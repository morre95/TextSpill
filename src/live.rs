//! Append-only live typing from consecutive audio segments.
//!
//! pw-record finalises WAV lengths only on exit. We snapshot actual PCM bytes
//! into a separate, valid WAV and ask the warm daemon about unconsumed audio.
//! A session token prevents delayed replies from leaking into another dictation.

use crate::{audio, clipboard, input, ipc, notify, paths::Paths, state};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const INTERVAL: Duration = Duration::from_millis(1000);
const BYTES_PER_SECOND: u64 = 32000;
const WINDOW_BYTES: u64 = 15 * BYTES_PER_SECOND;

#[derive(Serialize, Deserialize)]
pub struct Preview {
    pub text: String,
    pub language: Option<String>,
    pub provisional: bool,
    pub window_start_seconds: f64,
    pub audio_seconds: f64,
    pub error: Option<String>,
    #[serde(default)]
    pub consumed_bytes: u64,
    #[serde(default)]
    pub delivery_pending: bool,
}

/// Called with the command lock held, after the recorder has started.
pub fn start(paths: &Paths, recorder: i32) -> Result<()> {
    invalidate(paths)?;
    if std::env::var("TEXTSPILL_LIVE").as_deref() == Ok("0") {
        return Ok(());
    }
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let session = format!("{}-{stamp}", std::process::id());
    private_file(&paths.live_session())?.write_all(session.as_bytes())?;
    let log = private_file(&paths.live_log())?;
    if let Err(error) = Command::new(std::env::current_exe()?)
        .args(["live-worker", &session, &recorder.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .process_group(0)
        .spawn()
    {
        invalidate(paths)?;
        return Err(error).context("could not start live preview worker");
    }
    Ok(())
}

/// Called under the same lock as publication, before stop/cancel/new start.
pub fn invalidate(paths: &Paths) -> Result<()> {
    crate::remove_if_present(&paths.live_session())?;
    crate::remove_if_present(&paths.preview())
}

fn active(paths: &Paths, session: &str, recorder: i32) -> bool {
    fs::read_to_string(paths.live_session()).ok().as_deref() == Some(session)
        && state::is_running(recorder, audio::PROCESS_NAME)
}

pub fn worker(paths: &Paths, session: &str, recorder: i32) -> Result<()> {
    // The hidden CLI is still user input. Never allow path components here.
    if session.is_empty()
        || session.len() > 80
        || !session.bytes().all(|b| b.is_ascii_digit() || b == b'-')
    {
        bail!("invalid live session token");
    }
    let snapshot = paths.live_audio(session);
    let _cleanup = SnapshotCleanup(snapshot.clone());
    let mut progress = Preview {
        text: String::new(),
        language: None,
        provisional: false,
        window_start_seconds: 0.0,
        audio_seconds: 0.0,
        error: None,
        consumed_bytes: 0,
        delivery_pending: false,
    };
    let mut source = None;
    while active(paths, session, recorder) {
        std::thread::sleep(INTERVAL);
        if !active(paths, session, recorder) {
            break;
        }
        // Keep the descriptor: replacing recording.wav in a later session
        // cannot make this worker read that session's audio.
        if source.is_none() {
            source = File::open(paths.recording_wav()).ok();
        }
        let Some(source) = source.as_mut() else {
            continue;
        };
        let mut frame = match snapshot_pcm(source, progress.consumed_bytes) {
            Ok(Some(frame)) => frame,
            Ok(_) => continue,
            Err(error) => {
                publish_error(paths, session, recorder, &format!("{error:#}"))?;
                break;
            }
        };
        let Some(cut) = segment_boundary(&frame.pcm) else {
            continue;
        };
        frame.pcm.truncate(cut);
        frame.end = frame.start + cut as u64;
        write_wav(&snapshot, &frame.pcm)?;
        let response = match transcribe_segment(paths, &snapshot, &frame.pcm) {
            Ok(response) => response,
            Err(error) => {
                publish_error(paths, session, recorder, &format!("{error:#}"))?;
                break;
            }
        };
        // Inference runs without the command lock. Only publication holds it,
        // so stop/cancel can invalidate this session while inference is busy.
        let _lock = state::acquire_lock(paths)?;
        if !active(paths, session, recorder) {
            break;
        }
        if let Err(error) = deliver(paths, &mut progress, &frame, response) {
            progress.error = Some(format!("{error:#}"));
            publish(paths, &progress)?;
            notify::live("Live typing stopped; audio kept for recovery", &mut None);
            break;
        }
    }
    Ok(())
}

fn publish_error(paths: &Paths, session: &str, recorder: i32, error: &str) -> Result<()> {
    let _lock = state::acquire_lock(paths)?;
    if active(paths, session, recorder) {
        let mut progress = read(paths)?.unwrap_or(Preview {
            text: String::new(),
            language: None,
            provisional: true,
            window_start_seconds: 0.0,
            audio_seconds: 0.0,
            error: Some(error.to_owned()),
            consumed_bytes: 0,
            delivery_pending: false,
        });
        progress.error = Some(error.to_owned());
        publish(paths, &progress)?;
        // Use the bounded notification helper: preview failure is advisory.
        notify::live("Live preview unavailable; recording continues", &mut None);
    }
    tracing::warn!(%error, "live preview stopped; final transcription remains available");
    Ok(())
}

fn publish(paths: &Paths, preview: &Preview) -> Result<()> {
    let temporary = paths.preview().with_extension("tmp");
    private_file(&temporary)?.write_all(&serde_json::to_vec(preview)?)?;
    fs::rename(temporary, paths.preview())?;
    Ok(())
}

/// Prefer a short pause after at least 1.5 seconds, otherwise flush at 4 s.
/// Already emitted text is never selected, erased or rewritten.
fn segment_boundary(pcm: &[u8]) -> Option<usize> {
    const MIN: usize = 48000;
    const MAX: usize = 128000;
    const QUIET: usize = 5120; // 160 ms at 16 kHz, PCM16
    if pcm.len() < MIN {
        return None;
    }
    let limit = pcm.len().min(MAX);
    let mut quiet = 0;
    let mut boundary = None;
    for (index, sample) in pcm[..limit].as_chunks::<2>().0.iter().enumerate() {
        let value = i16::from_le_bytes([sample[0], sample[1]]);
        quiet = if value.unsigned_abs() < 250 {
            quiet + 2
        } else {
            0
        };
        let end = (index + 1) * 2;
        if end >= MIN && quiet >= QUIET {
            boundary = Some(end);
        }
    }
    boundary.or_else(|| (pcm.len() >= MAX).then_some(MAX))
}

fn transcribe_segment(
    paths: &Paths,
    snapshot: &std::path::Path,
    pcm: &[u8],
) -> Result<ipc::Transcription> {
    if pcm.iter().all(|b| *b == 0) {
        return Ok(ipc::Transcription {
            text: String::new(),
            language: None,
        });
    }
    ipc::preview(&paths.asr_socket(), snapshot)
}

fn deliver(
    paths: &Paths,
    progress: &mut Preview,
    frame: &Frame,
    result: ipc::Transcription,
) -> Result<()> {
    let text = crate::sanitize(&result.text);
    if !text.is_empty() {
        let addition = if progress.text.is_empty() {
            text.clone()
        } else {
            format!(" {text}")
        };
        clipboard::copy(&addition)?;
        // Persist intent before injecting input: if the process dies here,
        // stop must not automatically replay an ambiguously delivered segment.
        progress.delivery_pending = true;
        publish(paths, progress)?;
        if input::paste()?.is_none() {
            bail!("live typing requires a paste backend");
        }
        progress.text.push_str(&addition);
        progress.delivery_pending = false;
    }
    progress.consumed_bytes = frame.end;
    progress.window_start_seconds = frame.start as f64 / BYTES_PER_SECOND as f64;
    progress.audio_seconds = frame.end as f64 / BYTES_PER_SECOND as f64;
    progress.language = result.language;
    progress.error = None;
    publish(paths, progress)
}

/// The caller holds the command lock and has stopped recording. Only process
/// audio after the last committed segment, never replay the whole dictation.
pub fn finish(paths: &Paths, mut progress: Preview) -> Result<()> {
    if progress.delivery_pending {
        bail!("a live paste could not be confirmed; recording.wav is kept for manual recovery");
    }
    let snapshot = paths.live_audio("final");
    let _cleanup = SnapshotCleanup(snapshot.clone());
    let mut source = File::open(paths.recording_wav())?;
    while let Some(frame) = snapshot_pcm(&mut source, progress.consumed_bytes)? {
        write_wav(&snapshot, &frame.pcm)?;
        let result = transcribe_segment(paths, &snapshot, &frame.pcm)?;
        deliver(paths, &mut progress, &frame, result)?;
    }
    if !progress.text.is_empty() {
        // Full text remains available for manual reuse; do not paste it again.
        clipboard::copy(&progress.text)?;
    }
    notify::spilled(progress.language.as_deref(), &progress.text);
    crate::remove_if_present(&paths.recording_wav())?;
    crate::remove_if_present(&paths.preview())
}

pub fn read(paths: &Paths) -> Result<Option<Preview>> {
    if !matches!(state::current(paths)?, state::State::Recording { .. }) {
        return Ok(None);
    }
    match fs::read(paths.preview()) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn private_file(path: &std::path::Path) -> std::io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
}

struct SnapshotCleanup(std::path::PathBuf);
impl Drop for SnapshotCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct Frame {
    pcm: Vec<u8>,
    start: u64,
    end: u64,
}

/// Parse only headers, then seek straight to the bounded PCM tail. RIFF and
/// data lengths may be zero/stale while the recorder is still writing.
fn snapshot_pcm(source: &mut File, start: u64) -> Result<Option<Frame>> {
    source.rewind()?;
    let length = source.metadata()?.len();
    if length < 44 {
        return Ok(None);
    }
    let mut riff = [0; 12];
    source.read_exact(&mut riff)?;
    if &riff[..4] != b"RIFF" || &riff[8..] != b"WAVE" {
        bail!("capture is not a RIFF WAV");
    }
    let mut valid_format = false;
    loop {
        let position = source.stream_position()?;
        if position > 65536 {
            bail!("WAV header is too large");
        }
        if position + 8 > length {
            return Ok(None);
        }
        let mut chunk = [0; 8];
        source.read_exact(&mut chunk)?;
        let size = u32::from_le_bytes(chunk[4..8].try_into()?) as u64;
        let data = position + 8;
        if &chunk[..4] == b"data" {
            if !valid_format {
                bail!("capture requires 16 kHz mono PCM16 WAV");
            }
            let end = ((length - data) & !1).min(start + WINDOW_BYTES);
            if end <= start {
                return Ok(None);
            }
            source.seek(SeekFrom::Start(data + start))?;
            let mut pcm = vec![0; (end - start) as usize];
            source.read_exact(&mut pcm)?;
            return Ok(Some(Frame { pcm, start, end }));
        }
        if data + size > length {
            return Ok(None);
        }
        if &chunk[..4] == b"fmt " {
            if size < 16 {
                bail!("invalid WAV fmt chunk");
            }
            let mut fmt = [0; 16];
            source.read_exact(&mut fmt)?;
            valid_format = fmt[..2] == [1, 0]
                && fmt[2..4] == [1, 0]
                && u32::from_le_bytes(fmt[4..8].try_into()?) == 16000
                && fmt[12..14] == [2, 0]
                && fmt[14..16] == [16, 0];
        }
        source.seek(SeekFrom::Start(data + size + (size % 2)))?;
    }
}

fn write_wav(path: &std::path::Path, pcm: &[u8]) -> Result<()> {
    let mut wav = private_file(path)?;
    wav.write_all(b"RIFF")?;
    wav.write_all(&(36 + pcm.len() as u32).to_le_bytes())?;
    wav.write_all(b"WAVEfmt \x10\0\0\0\x01\0\x01\0")?;
    wav.write_all(&16000u32.to_le_bytes())?;
    wav.write_all(&32000u32.to_le_bytes())?;
    wav.write_all(b"\x02\0\x10\0data")?;
    wav.write_all(&(pcm.len() as u32).to_le_bytes())?;
    wav.write_all(pcm)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmenting_waits_for_a_pause_or_four_seconds() {
        let speech = [0xff, 0x20].repeat(64000);
        assert_eq!(segment_boundary(&speech[..32000]), None);
        assert_eq!(segment_boundary(&speech[..64000]), None);
        assert_eq!(segment_boundary(&speech), Some(128000));
        let mut paused = speech[..48000].to_vec();
        paused.extend(vec![0; 6400]);
        assert_eq!(segment_boundary(&paused), Some(paused.len()));
    }

    #[test]
    fn snapshots_ignore_stale_lengths_and_keep_only_the_recent_window() {
        let path = std::env::temp_dir().join(format!("textspill-live-{}.wav", std::process::id()));
        let _cleanup = SnapshotCleanup(path.clone());
        write_wav(&path, &vec![1; 20 * BYTES_PER_SECOND as usize]).unwrap();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(4)).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.seek(SeekFrom::Start(40)).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap();
        let frame = snapshot_pcm(&mut file, 5 * BYTES_PER_SECOND)
            .unwrap()
            .unwrap();
        assert_eq!(frame.start, 5 * BYTES_PER_SECOND);
        assert_eq!(frame.end, 20 * BYTES_PER_SECOND);
        assert_eq!(frame.pcm.len(), WINDOW_BYTES as usize);
        // Partial writes must not be mistaken for an entire corrupt recording.
        file.set_len(20).unwrap();
        assert!(snapshot_pcm(&mut file, 0).unwrap().is_none());
    }
}

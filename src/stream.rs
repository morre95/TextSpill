//! One detached worker owns the Deepgram connection and all transcript delivery.
//! Stop requests a drain under the command lock, then waits WITHOUT that lock.
use crate::{
    audio, clipboard,
    config::{self, SessionConfig},
    deepgram, live, notify,
    paths::Paths,
    state,
    transcription::Transcription,
};
use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::{fs::OpenOptionsExt, process::CommandExt},
    process::{Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream,
    tungstenite::{self, Message, client::IntoClientRequest},
};

const FINISH_TIMEOUT: Duration = Duration::from_secs(30);
type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Phase {
    Recording,
    Finishing,
    Done,
    Error,
}

#[derive(Serialize, Deserialize)]
struct StreamState {
    session: String,
    worker_pid: i32,
    phase: Phase,
    error: Option<String>,
}

fn read_state(paths: &Paths) -> Result<StreamState> {
    serde_json::from_slice(&fs::read(paths.stream_state())?)
        .context("invalid streaming session state")
}

fn matches_session(paths: &Paths, session: &str) -> bool {
    fs::read_to_string(paths.live_session()).ok().as_deref() == Some(session)
}

fn empty_preview(language: &str) -> live::Preview {
    live::Preview {
        text: String::new(),
        language: Some(language.into()),
        provisional: false,
        window_start_seconds: 0.,
        audio_seconds: 0.,
        error: None,
        consumed_bytes: 0,
        delivery_pending: false,
    }
}

/// Caller holds command lock, including while publishing the child's identity.
pub fn start(paths: &Paths, recorder: i32, language: &str) -> Result<()> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let session = format!("{}-{stamp}", std::process::id());
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(paths.live_session())?
        .write_all(session.as_bytes())?;
    live::publish(paths, &empty_preview(language))?;
    let log = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(paths.live_log())?;
    let child = Command::new(std::env::current_exe()?)
        .args(["stream-worker", &session, &recorder.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .process_group(0)
        .spawn()
        .context("could not start Deepgram worker")?;
    config::write_json(
        &paths.stream_state(),
        &StreamState {
            session,
            worker_pid: child.id() as i32,
            phase: Phase::Recording,
            error: None,
        },
    )
}

/// Called after recorder exit, while still holding the command lock.
pub fn request_finish(paths: &Paths) -> Result<String> {
    let mut status = read_state(paths).context("Deepgram worker is unavailable; audio kept")?;
    if !matches_session(paths, &status.session) {
        bail!("stream session is no longer active; audio kept");
    }
    if status.phase == Phase::Recording {
        status.phase = Phase::Finishing;
    }
    config::write_json(&paths.stream_state(), &status)?;
    Ok(status.session)
}

/// Stop owns transcribing.pid while waiting; the worker only holds short delivery locks.
pub fn finish(paths: &Paths, session: &str) -> Result<()> {
    let deadline = Instant::now() + FINISH_TIMEOUT;
    loop {
        {
            let _lock = state::acquire_lock(paths)?;
            if !matches_session(paths, session) {
                return Ok(());
            } // canceled
            let mut status = read_state(paths)?;
            if status.session != session {
                return Ok(());
            }
            if status.phase == Phase::Done {
                let progress: live::Preview = serde_json::from_slice(&fs::read(paths.preview())?)?;
                if !progress.text.is_empty() {
                    clipboard::copy(&progress.text)?;
                }
                notify::spilled(progress.language.as_deref(), &progress.text);
                crate::remove_if_present(&paths.recording_wav())?;
                live::invalidate(paths)?;
                crate::remove_if_present(&paths.stream_state())?;
                return Ok(());
            }
            let error = if status.phase == Phase::Error {
                status
                    .error
                    .clone()
                    .unwrap_or_else(|| "Deepgram worker failed".into())
            } else if !state::is_running(status.worker_pid, "textspill") {
                "Deepgram worker exited before completing the stream".into()
            } else if Instant::now() >= deadline {
                "Deepgram stream did not finish within 30s".into()
            } else {
                String::new()
            };
            if !error.is_empty() {
                // Invalidation and publication share a lock: no paste can race this failure.
                crate::remove_if_present(&paths.live_session())?;
                status.phase = Phase::Error;
                status.error = Some(error.clone());
                config::write_json(&paths.stream_state(), &status)?;
                let mut progress: live::Preview =
                    serde_json::from_slice(&fs::read(paths.preview())?)?;
                progress.error = Some(error.clone());
                live::publish(paths, &progress)?;
                bail!("{error}; recording.wav and preview.json kept for recovery");
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

pub fn worker(paths: &Paths, session: &str, recorder: i32) -> Result<()> {
    let (settings, mut source) = {
        let _lock = state::acquire_lock(paths)?;
        if !matches_session(paths, session) {
            return Ok(());
        }
        let status = read_state(paths)?;
        if status.session != session || status.worker_pid != std::process::id() as i32 {
            bail!("invalid streaming worker identity");
        }
        (
            SessionConfig::load(paths)?,
            File::open(paths.recording_wav())?,
        )
    };
    let mut progress = empty_preview(&settings.config.deepgram_language);
    let result = (|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(run_stream(
            paths,
            session,
            recorder,
            &settings,
            &mut source,
            &mut progress,
            &deepgram::endpoint(true),
        ))
    })();
    let _lock = state::acquire_lock(paths)?;
    if !matches_session(paths, session) {
        return Ok(());
    }
    let mut status = read_state(paths)?;
    match result {
        Ok(()) => {
            status.phase = Phase::Done;
        }
        Err(error) => {
            // Errors contain only our messages, never remote payloads/headers.
            let message = format!("{error:#}");
            progress.error = Some(message.clone());
            live::publish(paths, &progress)?;
            status.phase = Phase::Error;
            status.error = Some(message);
            notify::live(
                "Deepgram live typing stopped; audio kept for recovery",
                &mut None,
            );
        }
    }
    config::write_json(&paths.stream_state(), &status)
}

async fn send(socket: &mut Socket, message: Message) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(5), socket.send(message))
        .await
        .context("Deepgram send timed out")?
        .map_err(|_| anyhow::anyhow!("Deepgram connection failed while sending audio"))
}

async fn run_stream(
    paths: &Paths,
    session: &str,
    recorder: i32,
    settings: &SessionConfig,
    source: &mut File,
    progress: &mut live::Preview,
    endpoint: &str,
) -> Result<()> {
    settings.config.validate_language()?;
    let url = format!(
        "{endpoint}?model=nova-3&language={}&encoding=linear16&sample_rate=16000&channels=1&punctuate=true&interim_results=true&endpointing=300",
        settings.config.deepgram_language
    );
    let mut request = url.into_client_request()?;
    let mut header =
        tungstenite::http::HeaderValue::from_str(&format!("Token {}", config::api_key()?))
            .map_err(|_| anyhow::anyhow!("invalid Deepgram API key"))?;
    header.set_sensitive(true);
    request.headers_mut().insert("Authorization", header);
    let ws_config = tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(1_048_576))
        .max_frame_size(Some(1_048_576));
    let (mut socket, _) = tokio::time::timeout(
        Duration::from_secs(10),
        tokio_tungstenite::connect_async_with_config(request, Some(ws_config), true),
    )
    .await
    .context("Deepgram connection timed out")?
    .map_err(|e| match e {
        tungstenite::Error::Http(response) => {
            anyhow::anyhow!(deepgram::status_error(response.status().as_u16()))
        }
        _ => anyhow::anyhow!("Deepgram connection failed; check network access"),
    })?;
    let mut sent = 0u64;
    let mut last_send = Instant::now();
    let mut closing: Option<Instant> = None;
    let mut metadata = false;
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        if !matches_session(paths, session) {
            return Ok(());
        }
        tokio::select! {
            _ = tick.tick() => {
                if closing.is_some_and(|at| at.elapsed() >= FINISH_TIMEOUT) {
                    bail!("Deepgram final results timed out");
                }
                if closing.is_none() {
                    let status = read_state(paths)?;
                    if status.session != session { return Ok(()); }
                    // Only explicit stop drains a stream. A crashed recorder is a failure.
                    if status.phase == Phase::Recording && !state::is_running(recorder, audio::PROCESS_NAME) {
                        // Stop may be finalizing the WAV while holding this lock.
                        let _lock = state::acquire_lock(paths)?;
                        if read_state(paths)?.phase == Phase::Recording {
                            bail!("audio recorder exited unexpectedly");
                        }
                        continue;
                    }
                    if let Some(frame) = live::snapshot_pcm(source, sent)? {
                        // Catch up after connection setup, but bound each iteration to 1 s.
                        for chunk in frame.pcm[..frame.pcm.len().min(32000)].chunks(3200) {
                            if !matches_session(paths, session) { return Ok(()); }
                            send(&mut socket, Message::Binary(chunk.to_vec().into())).await?;
                            sent += chunk.len() as u64;
                        }
                        last_send = Instant::now();
                    } else if status.phase == Phase::Finishing {
                        send(&mut socket, Message::Text(r#"{"type":"CloseStream"}"#.into())).await?;
                        closing = Some(Instant::now());
                    } else if last_send.elapsed() >= Duration::from_secs(3) {
                        send(&mut socket, Message::Text(r#"{"type":"KeepAlive"}"#.into())).await?;
                        last_send = Instant::now();
                    }
                }
            }
            message = socket.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        let value: serde_json::Value = serde_json::from_str(&text).context("invalid Deepgram streaming JSON")?;
                        match value.get("type").and_then(|v| v.as_str()) {
                            Some("Results") => {
                                let result: deepgram::StreamResult = serde_json::from_value(value).context("invalid Deepgram streaming result")?;
                                if result.is_final {
                                    let _lock = state::acquire_lock(paths)?;
                                    if !matches_session(paths, session) { return Ok(()); }
                                    commit_result(paths, progress, result, sent)?;
                                }
                            }
                            Some("Metadata") if closing.is_some() => metadata = true,
                            Some("Error") => bail!("Deepgram streaming API reported an error; audio kept"),
                            _ => {}
                        }
                    }
                    Some(Ok(Message::Ping(payload))) => send(&mut socket, Message::Pong(payload)).await?,
                    Some(Ok(Message::Close(_))) | None => {
                        if closing.is_some() && metadata { return Ok(()); }
                        bail!("Deepgram stream closed before final results were confirmed");
                    }
                    Some(Err(_)) => bail!("Deepgram stream disconnected; audio kept"),
                    _ => {}
                }
            }
        }
    }
}

fn commit_result(
    paths: &Paths,
    progress: &mut live::Preview,
    result: deepgram::StreamResult,
    sent: u64,
) -> Result<()> {
    if !result.is_final {
        return Ok(());
    }
    if !result.start.is_finite()
        || !result.duration.is_finite()
        || result.start < 0.
        || result.duration < 0.
    {
        bail!("invalid Deepgram result timing");
    }
    let end = (((result.start + result.duration) * 16000.).round() as u64).saturating_mul(2);
    if end <= progress.consumed_bytes {
        return Ok(());
    } // duplicate finalized interval
    if end > sent + 3200 {
        bail!("Deepgram result exceeds submitted audio");
    }
    let alternative = result
        .channel
        .alternatives
        .into_iter()
        .next()
        .context("Deepgram result has no alternative")?;
    let frame = live::Frame {
        pcm: Vec::new(),
        start: progress.consumed_bytes,
        end: end.min(sent),
    };
    live::deliver(
        paths,
        progress,
        &frame,
        Transcription {
            text: alternative.transcript,
            language: progress.language.clone(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interim_and_duplicate_results_never_reach_clipboard() {
        let paths = Paths::for_test(std::env::temp_dir());
        let mut progress = empty_preview("sv");
        progress.consumed_bytes = 32000;
        for is_final in [false, true] {
            let result = serde_json::from_value(
                serde_json::json!({"is_final":is_final,"start":0,"duration":1,
                "channel":{"alternatives":[{"transcript":"must not paste"}]}}),
            )
            .unwrap();
            commit_result(&paths, &mut progress, result, 32000).unwrap();
        }
        assert!(progress.text.is_empty());
        assert_eq!(progress.consumed_bytes, 32000);
    }
}

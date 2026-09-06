//! Unix-socket client for the Qwen3-ASR daemon.
//!
//! The protocol is one line of JSON in each direction. It is deliberately
//! trivial so that it can be exercised by hand:
//!
//! ```text
//! $ printf '{"audio_path":"/run/user/1000/textspill/recording.wav"}\n' \
//!     | socat - UNIX-CONNECT:/run/user/1000/textspill/asr.sock
//! ```
//!
//! Every socket operation has a deadline: a hung or half-dead daemon must never
//! leave the hotkey path blocked forever.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

const CONNECT_HINT: &str = "start it with: systemctl --user start textspill-asr.service";

/// The daemon is local; a write that blocks this long is a dead peer.
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// A response line is small; this only bounds a misbehaving peer.
const MAX_RESPONSE_BYTES: u64 = 1 << 20;

#[derive(Serialize)]
struct Request<'a> {
    audio_path: &'a str,
}

/// Either `{"text": ..., "language": ...}` or `{"error": ...}`.
#[derive(Deserialize)]
struct Response {
    text: Option<String>,
    language: Option<String>,
    error: Option<String>,
}

/// A successful transcription.
pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
}

/// Sends `wav` to the daemon at `socket` and waits for the transcription.
pub fn transcribe(socket: &Path, wav: &Path, read_timeout: Duration) -> Result<Transcription> {
    let audio_path = wav
        .to_str()
        .ok_or_else(|| anyhow!("recording path is not valid UTF-8"))?;

    let stream = UnixStream::connect(socket).map_err(|e| {
        anyhow!(
            "ASR daemon is not reachable on {} ({e}); {CONNECT_HINT}",
            socket.display()
        )
    })?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    stream.set_read_timeout(Some(read_timeout))?;

    let started = Instant::now();
    let mut writer = &stream;
    let mut line = serde_json::to_vec(&Request { audio_path })?;
    line.push(b'\n');
    writer
        .write_all(&line)
        .context("failed to send request to ASR daemon")?;
    writer.flush().ok();

    let mut reply = String::new();
    BufReader::new((&stream).take(MAX_RESPONSE_BYTES))
        .read_line(&mut reply)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => anyhow!(
                "ASR daemon did not answer within {}s",
                read_timeout.as_secs()
            ),
            _ => anyhow!("failed to read from ASR daemon: {e}"),
        })?;

    if reply.trim().is_empty() {
        bail!("ASR daemon closed the connection without replying");
    }

    let response: Response = serde_json::from_str(reply.trim())
        .with_context(|| format!("ASR daemon sent invalid JSON: {:?}", truncate(&reply, 200)))?;

    if let Some(error) = response.error {
        bail!("ASR daemon reported: {error}");
    }
    let text = response
        .text
        .ok_or_else(|| anyhow!("ASR daemon reply had neither `text` nor `error`"))?;

    tracing::info!(
        latency_ms = started.elapsed().as_millis(),
        language = response.language.as_deref().unwrap_or("unknown"),
        chars = text.chars().count(),
        "transcription received"
    );

    Ok(Transcription {
        text,
        language: response.language,
    })
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

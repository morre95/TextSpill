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
    preview: bool,
}

/// Either `{"text": ..., "language": ...}` or `{"error": ...}`.
#[derive(Deserialize)]
struct Response {
    text: Option<String>,
    language: Option<String>,
    error: Option<String>,
}

/// A successful transcription.
#[derive(Debug)]
pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
}

/// Sends `wav` to the daemon at `socket` and waits for the transcription.
pub fn transcribe(socket: &Path, wav: &Path, read_timeout: Duration) -> Result<Transcription> {
    request(socket, wav, read_timeout, false)
}

pub fn preview(socket: &Path, wav: &Path) -> Result<Transcription> {
    request(socket, wav, Duration::from_secs(30), true)
}

fn request(
    socket: &Path,
    wav: &Path,
    read_timeout: Duration,
    preview: bool,
) -> Result<Transcription> {
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
    tracing::debug!(socket = %socket.display(), "connected to ASR daemon");

    let started = Instant::now();
    let mut writer = &stream;
    let mut line = serde_json::to_vec(&Request {
        audio_path,
        preview,
    })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;
    use std::thread;

    /// Serves one connection with a scripted reply, then closes.
    ///
    /// Replaying the daemon's side over a real socket is what makes these tests
    /// worth having: they exercise connect, framing, timeouts and parsing
    /// exactly as the hotkey path does.
    struct FakeDaemon {
        socket: PathBuf,
        _dir: PathBuf,
    }

    impl FakeDaemon {
        fn new(name: &str, reply: Option<&'static [u8]>) -> Self {
            let dir =
                std::env::temp_dir().join(format!("textspill-ipc-{}-{name}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let socket = dir.join("asr.sock");
            let _ = std::fs::remove_file(&socket);

            let listener = UnixListener::bind(&socket).unwrap();
            thread::spawn(move || {
                let Ok((mut conn, _)) = listener.accept() else {
                    return;
                };
                let mut request = String::new();
                BufReader::new(conn.try_clone().unwrap())
                    .read_line(&mut request)
                    .ok();
                match reply {
                    Some(bytes) => {
                        conn.write_all(bytes).ok();
                    }
                    // Hold the connection open without answering, to drive the
                    // client into its read timeout.
                    None => thread::sleep(Duration::from_secs(30)),
                }
            });

            Self { socket, _dir: dir }
        }

        fn ask(&self) -> Result<Transcription> {
            transcribe(
                &self.socket,
                Path::new("/run/user/1000/textspill/recording.wav"),
                Duration::from_millis(300),
            )
        }
    }

    #[test]
    fn parses_a_successful_reply() {
        let daemon = FakeDaemon::new(
            "ok",
            Some(b"{\"text\":\"hej v\xc3\xa4rlden\",\"language\":\"Swedish\"}\n"),
        );
        let result = daemon.ask().expect("should succeed");
        assert_eq!(result.text, "hej världen");
        assert_eq!(result.language.as_deref(), Some("Swedish"));
    }

    #[test]
    fn accepts_a_reply_without_a_language() {
        let daemon = FakeDaemon::new("nolang", Some(b"{\"text\":\"hello\"}\n"));
        let result = daemon.ask().expect("language is optional");
        assert_eq!(result.text, "hello");
        assert_eq!(result.language, None);
    }

    #[test]
    fn surfaces_a_daemon_error() {
        let daemon = FakeDaemon::new("err", Some(b"{\"error\":\"CUDA out of memory\"}\n"));
        let e = daemon.ask().expect_err("an error reply must not succeed");
        assert!(e.to_string().contains("CUDA out of memory"), "{e}");
    }

    #[test]
    fn rejects_invalid_json() {
        let daemon = FakeDaemon::new("garbage", Some(b"<html>502</html>\n"));
        let e = daemon.ask().expect_err("garbage must not parse");
        assert!(e.to_string().contains("invalid JSON"), "{e}");
    }

    #[test]
    fn rejects_a_reply_with_neither_text_nor_error() {
        let daemon = FakeDaemon::new("neither", Some(b"{\"language\":\"Swedish\"}\n"));
        let e = daemon.ask().expect_err("an empty reply must not succeed");
        assert!(e.to_string().contains("neither"), "{e}");
    }

    #[test]
    fn detects_a_daemon_that_closes_without_replying() {
        let daemon = FakeDaemon::new("die", Some(b""));
        let e = daemon.ask().expect_err("a silent close must not succeed");
        assert!(e.to_string().contains("without replying"), "{e}");
    }

    #[test]
    fn times_out_instead_of_hanging_forever() {
        let daemon = FakeDaemon::new("hang", None);
        let started = Instant::now();
        let e = daemon.ask().expect_err("a hung daemon must not block");
        assert!(e.to_string().contains("did not answer"), "{e}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "returned after {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn reports_a_missing_daemon_with_a_hint() {
        let e = transcribe(
            Path::new("/nonexistent/textspill/asr.sock"),
            Path::new("/tmp/recording.wav"),
            Duration::from_millis(300),
        )
        .expect_err("no socket means no daemon");
        let message = e.to_string();
        assert!(message.contains("not reachable"), "{message}");
        assert!(message.contains("systemctl --user start"), "{message}");
    }
}

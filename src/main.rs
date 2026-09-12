//! TextSpill — system-wide voice-to-text for Linux.
//! Speak, transcribe, spill text wherever your cursor is.
//!
//! Records and pastes through either the warm local Qwen3-ASR daemon or Deepgram.

mod audio;
mod clipboard;
mod config;
mod deepgram;
mod input;
mod ipc;
mod live;
mod notify;
mod paths;
mod state;
mod stream;
mod transcription;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use paths::Paths;
use state::State;
use std::path::Path;
use std::process::ExitCode;

/// A WAV shorter than this holds no speech worth sending.
/// 16 kHz × 1 channel × 2 bytes ≈ 32 kB per second, plus a 44-byte header.
const MIN_AUDIO_BYTES: u64 = 44 + 32_000 / 4;

#[derive(Parser)]
#[command(
    name = "textspill",
    version,
    about = "System-wide voice-to-text for Linux. Speak, transcribe, spill text wherever your cursor is."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the saved backend, or select one without changing other settings.
    Configure {
        #[arg(long, value_enum)]
        backend: Option<config::Backend>,
    },
    /// Start recording, or stop and spill the transcription. Bind this to a hotkey.
    Toggle,
    /// Start recording.
    Start,
    /// Stop recording, transcribe, copy and paste.
    Stop,
    /// Stop recording and throw the audio away.
    Cancel,
    /// Print `idle`, `recording` or `transcribing`.
    Status,
    /// Print the latest provisional transcription (JSON with --json).
    Preview {
        #[arg(long)]
        json: bool,
    },
    #[command(hide = true)]
    LiveWorker { session: String, recorder: i32 },
    #[command(hide = true)]
    StreamWorker { session: String, recorder: i32 },
}

fn main() -> ExitCode {
    init_tracing();
    let cli = Cli::parse();

    match run(cli.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let message = format!("{e:#}");
            tracing::error!(error = %message, "command failed");
            eprintln!("textspill: {message}");
            notify::error(&message);
            ExitCode::FAILURE
        }
    }
}

/// `RUST_LOG=debug textspill toggle` turns on the detailed trace.
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
}

fn run(command: Command) -> Result<()> {
    if let Command::Configure { backend } = command {
        return config::configure(backend);
    }
    let paths = Paths::new()?;
    match command {
        Command::Configure { .. } => unreachable!(),
        Command::StreamWorker { session, recorder } => stream::worker(&paths, &session, recorder),
        Command::LiveWorker { session, recorder } => live::worker(&paths, &session, recorder),
        Command::Preview { json } => {
            let preview = live::read(&paths)?;
            if json {
                println!("{}", serde_json::to_string(&preview)?);
            } else if let Some(preview) = preview {
                if let Some(error) = preview.error {
                    anyhow::bail!("{error}");
                }
                println!("{}", preview.text);
            }
            Ok(())
        }
        // Read-only, and must never block on a running transcription.
        Command::Status => {
            println!("{}", state::current(&paths)?);
            Ok(())
        }
        Command::Toggle | Command::Start | Command::Stop | Command::Cancel => {
            let lock = state::acquire_lock(&paths)?;
            match command {
                Command::Toggle if matches!(state::current(&paths)?, State::Recording { .. }) => {
                    cmd_stop(&paths, lock)
                }
                Command::Toggle | Command::Start => cmd_start(&paths),
                Command::Stop => cmd_stop(&paths, lock),
                Command::Cancel => cmd_cancel(&paths),
                _ => unreachable!(),
            }
        }
    }
}

fn cmd_start(paths: &Paths) -> Result<()> {
    match state::current(paths)? {
        State::Recording { .. } => {
            notify::info("Already recording");
            return Ok(());
        }
        State::Transcribing => {
            notify::info("Still transcribing");
            return Ok(());
        }
        State::Idle => {}
    }

    // Validate credentials before clearing audio retained from an earlier failure.
    let settings = config::SessionConfig {
        config: config::Config::effective()?,
        live: std::env::var("TEXTSPILL_LIVE").as_deref() != Ok("0"),
    };
    config::write_json(&paths.session_config(), &settings)?;
    let wav = paths.recording_wav();
    let log = paths.capture_log();
    live::invalidate(paths)?;
    remove_if_present(&paths.stream_state())?;
    // Drop any recording kept from a previous failure before overwriting it.
    remove_if_present(&wav)?;

    let pid = audio::start(&wav, &log)?;
    if let Err(error) = state::write_pid(&paths.recording_pid(), pid) {
        let _ = audio::stop(pid);
        return Err(error);
    }
    notify::recording();

    // Verified after notifying: the user hears no delay, but a recorder that
    // died on startup must not leave a PID file claiming we are recording.
    if let Err(e) = audio::confirm_started(pid, &log) {
        state::remove_pid(&paths.recording_pid())?;
        return Err(e);
    }
    if settings.config.backend == config::Backend::Deepgram && settings.live {
        if let Err(error) = stream::start(paths, pid, &settings.config.deepgram_language) {
            let _ = audio::stop(pid);
            state::remove_pid(&paths.recording_pid())?;
            remove_if_present(&paths.live_session())?;
            return Err(error);
        }
    } else if let Err(error) = live::start(paths, pid) {
        tracing::warn!(%error, "live preview unavailable; recording continues");
    }
    Ok(())
}

fn cmd_stop(paths: &Paths, lock: state::Lock) -> Result<()> {
    let State::Recording { pid } = state::current(paths)? else {
        notify::info("Not recording");
        return Ok(());
    };

    let settings = config::SessionConfig::load(paths)?;
    let streaming = settings.config.backend == config::Backend::Deepgram && settings.live;
    let live_progress = live::read(paths)?;
    if !streaming {
        remove_if_present(&paths.live_session())?;
    }
    audio::stop(pid)?;
    state::remove_pid(&paths.recording_pid())?;

    // Publish `transcribing` for `textspill status`. The guard clears it even
    // if the ASR call fails, and a crash here reads as stale state next time.
    state::write_pid(&paths.transcribing_pid(), std::process::id() as i32)?;
    let _transcribing = ClearOnDrop(paths.transcribing_pid());
    notify::transcribing();

    if streaming {
        let session = stream::request_finish(paths)?;
        drop(lock);
        return stream::finish(paths, &session);
    }

    if let Some(progress) = live_progress {
        return live::finish(paths, progress);
    }

    let wav = paths.recording_wav();
    let bytes = std::fs::metadata(&wav)
        .with_context(|| format!("no recording at {}", wav.display()))?
        .len();
    if bytes < MIN_AUDIO_BYTES {
        tracing::warn!(bytes, "recording too short to transcribe");
        remove_if_present(&wav)?;
        notify::info("Nothing recorded");
        return Ok(());
    }

    // On any failure below, `recording.wav` is deliberately left in place so
    // the dictation can be retried or recovered by hand.
    let result = transcription::transcribe(paths, &settings.config, &wav)?;
    let text = sanitize(&result.text);
    if text.is_empty() {
        tracing::info!("ASR returned no speech");
        remove_if_present(&wav)?;
        notify::info("No speech detected");
        return Ok(());
    }

    clipboard::copy(&text)?;
    // From here the text is safe on the clipboard, so a failed keystroke is a
    // notification, not an error: the user can still paste it themselves.
    match input::paste() {
        Ok(_) => notify::spilled(result.language.as_deref(), &text),
        Err(e) => {
            tracing::warn!(error = %format!("{e:#}"), "paste failed");
            notify::clipboard_only(&format!("{e}"));
        }
    }

    remove_if_present(&wav)?;
    Ok(())
}

fn cmd_cancel(paths: &Paths) -> Result<()> {
    let current = state::current(paths)?;
    let cancel_stream = matches!(current, State::Transcribing) && paths.stream_state().exists();
    live::invalidate(paths)?;
    if cancel_stream {
        remove_if_present(&paths.recording_wav())?;
        remove_if_present(&paths.stream_state())?;
        notify::cancelled();
        return Ok(());
    }
    let State::Recording { pid } = current else {
        notify::info("Not recording");
        return Ok(());
    };
    audio::stop(pid)?;
    state::remove_pid(&paths.recording_pid())?;
    remove_if_present(&paths.recording_wav())?;
    tracing::info!("recording cancelled");
    notify::cancelled();
    Ok(())
}

/// Flattens the transcription into a single line.
///
/// A newline on the clipboard becomes an Enter keypress when pasted into a
/// shell or a CLI agent, which would run the command or send the prompt before
/// the user has read it. TextSpill never does that.
fn sanitize(text: &str) -> String {
    text.split(['\n', '\r'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn remove_if_present(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("failed to remove {}", path.display())),
    }
}

/// Removes a PID file when the enclosing scope ends, on success or error.
struct ClearOnDrop(std::path::PathBuf);

impl Drop for ClearOnDrop {
    fn drop(&mut self) {
        if let Err(e) = state::remove_pid(&self.0) {
            tracing::warn!(error = %e, "failed to clear state file");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize;

    #[test]
    fn sanitize_removes_every_line_break() {
        assert_eq!(sanitize("rm -rf /\nyes\r\n"), "rm -rf / yes");
        assert_eq!(sanitize("  hello  "), "hello");
        assert_eq!(sanitize("\n\n"), "");
        assert_eq!(sanitize("en mening"), "en mening");
    }
}

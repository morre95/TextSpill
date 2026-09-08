//! TextSpill — system-wide voice-to-text for Linux.
//! Speak, transcribe, spill text wherever your cursor is.
//!
//! The hotkey path is deliberately thin. All the expensive work — loading
//! Qwen3-ASR — happens once, in `textspill-asr.service`; this binary only
//! records, asks the warm daemon, and types the answer.

mod audio;
mod clipboard;
mod input;
mod ipc;
mod live;
mod notify;
mod paths;
mod state;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use paths::Paths;
use state::State;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

/// Upper bound on how long we wait for Qwen3-ASR. Generous, because the very
/// first request after the daemon starts also warms CUDA kernels.
const TRANSCRIBE_TIMEOUT: Duration = Duration::from_secs(120);

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
    let paths = Paths::new()?;
    match command {
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
        Command::Toggle => with_lock(&paths, cmd_toggle),
        Command::Start => with_lock(&paths, cmd_start),
        Command::Stop => with_lock(&paths, cmd_stop),
        Command::Cancel => with_lock(&paths, cmd_cancel),
    }
}

/// Runs a state-changing command under the runtime lock, so that two hotkey
/// presses cannot both observe `idle` and both start a recorder.
fn with_lock(paths: &Paths, action: fn(&Paths) -> Result<()>) -> Result<()> {
    let _lock = state::acquire_lock(paths)?;
    action(paths)
}

fn cmd_toggle(paths: &Paths) -> Result<()> {
    match state::current(paths)? {
        State::Recording { .. } => cmd_stop(paths),
        _ => cmd_start(paths),
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

    let wav = paths.recording_wav();
    let log = paths.capture_log();
    live::invalidate(paths)?;
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
    if let Err(error) = live::start(paths, pid) {
        tracing::warn!(%error, "live preview unavailable; recording continues");
    }
    Ok(())
}

fn cmd_stop(paths: &Paths) -> Result<()> {
    let State::Recording { pid } = state::current(paths)? else {
        notify::info("Not recording");
        return Ok(());
    };

    let live_progress = live::read(paths)?;
    remove_if_present(&paths.live_session())?;
    audio::stop(pid)?;
    state::remove_pid(&paths.recording_pid())?;

    // Publish `transcribing` for `textspill status`. The guard clears it even
    // if the ASR call fails, and a crash here reads as stale state next time.
    state::write_pid(&paths.transcribing_pid(), std::process::id() as i32)?;
    let _transcribing = ClearOnDrop(paths.transcribing_pid());
    notify::transcribing();

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
    let result = ipc::transcribe(&paths.asr_socket(), &wav, TRANSCRIBE_TIMEOUT)?;
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
    live::invalidate(paths)?;
    let State::Recording { pid } = state::current(paths)? else {
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

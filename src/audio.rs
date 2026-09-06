//! Microphone capture.
//!
//! The first prototype shells out to `pw-record`, which already handles device
//! selection, resampling and WAV muxing, and finalises the RIFF header on
//! SIGINT. Everything PipeWire-specific is confined to this module so that a
//! native capture backend can replace it without touching the rest of TextSpill.

use anyhow::{Context, Result, bail};
use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The executable we spawn, and the `comm` we expect to find in `/proc`.
pub const PROCESS_NAME: &str = "pw-record";

/// What Qwen3-ASR wants: 16 kHz mono 16-bit PCM.
const SAMPLE_RATE: &str = "16000";
const CHANNELS: &str = "1";
const SAMPLE_FORMAT: &str = "s16";

/// How long a stopped recorder gets to flush the WAV header before SIGKILL.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How long we wait to notice that capture failed to come up at all.
const STARTUP_CHECK: Duration = Duration::from_millis(150);

/// Spawns a detached recorder writing `wav`, returning its PID.
///
/// The child gets its own process group so that a Ctrl-C in the terminal that
/// launched `textspill` does not kill the recording, and it outlives us: the
/// hotkey path must return immediately.
pub fn start(wav: &Path, log: &Path) -> Result<i32> {
    use std::os::unix::process::CommandExt;

    let stderr = File::create(log)
        .with_context(|| format!("failed to create capture log {}", log.display()))?;

    let child = Command::new(PROCESS_NAME)
        .arg("--rate")
        .arg(SAMPLE_RATE)
        .arg("--channels")
        .arg(CHANNELS)
        .arg("--format")
        .arg(SAMPLE_FORMAT)
        .arg("--")
        .arg(wav)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr)
        .process_group(0)
        .spawn()
        .with_context(|| format!("failed to start {PROCESS_NAME} (is pipewire installed?)"))?;

    let pid = child.id() as i32;
    tracing::info!(pid, wav = %wav.display(), "recording started");
    Ok(pid)
}

/// Returns an error if the recorder died right after starting.
///
/// Called *after* the user has been told that recording began, so the check
/// costs nothing in perceived latency but still prevents a PID file that points
/// at a process which never captured a sample.
pub fn confirm_started(pid: i32, log: &Path) -> Result<()> {
    std::thread::sleep(STARTUP_CHECK);
    if crate::state::is_running(pid, PROCESS_NAME) {
        return Ok(());
    }
    let reason = std::fs::read_to_string(log).unwrap_or_default();
    let reason = reason.trim();
    if reason.is_empty() {
        bail!("{PROCESS_NAME} exited immediately");
    }
    bail!("{PROCESS_NAME} exited immediately: {reason}");
}

/// Stops the recorder and waits for it to close the WAV file.
///
/// SIGINT makes `pw-record` write a valid RIFF header before exiting; SIGKILL
/// is only a backstop, and leaves whatever was flushed to disk.
pub fn stop(pid: i32) -> Result<()> {
    signal(pid, Signal::SIGINT)?;
    if wait_for_exit(pid, SHUTDOWN_TIMEOUT) {
        tracing::info!(pid, "recording stopped");
        return Ok(());
    }

    tracing::warn!(pid, "recorder ignored SIGINT, sending SIGKILL");
    signal(pid, Signal::SIGKILL)?;
    if wait_for_exit(pid, SHUTDOWN_TIMEOUT) {
        Ok(())
    } else {
        bail!("{PROCESS_NAME} (pid {pid}) would not exit")
    }
}

fn signal(pid: i32, sig: Signal) -> Result<()> {
    // Re-check the process identity immediately before signalling: `pid` came
    // from a file on disk and must never be aimed at a recycled PID.
    if !crate::state::is_running(pid, PROCESS_NAME) {
        return Ok(());
    }
    match kill(Pid::from_raw(pid), sig) {
        Ok(()) | Err(nix::errno::Errno::ESRCH) => Ok(()),
        Err(e) => Err(e).with_context(|| format!("failed to send {sig} to pid {pid}")),
    }
}

fn wait_for_exit(pid: i32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !crate::state::is_running(pid, PROCESS_NAME) {
            return true;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    !crate::state::is_running(pid, PROCESS_NAME)
}

//! Recording state, derived from PID files under the runtime dir.
//!
//! State is not stored as a value that can go out of sync with reality; it is
//! *derived* by asking `/proc` whether the recorded PIDs are still alive. A
//! crashed `pw-record` therefore reads as `idle`, and the stale file is removed
//! as a side effect of looking.

use crate::paths::Paths;
use anyhow::{Context, Result, bail};
use std::fmt;
use std::fs::{self, File, TryLockError};
use std::path::Path;

/// What TextSpill is doing right now.
#[derive(Debug)]
pub enum State {
    Idle,
    /// `pw-record` is capturing.
    Recording {
        pid: i32,
    },
    /// Another `textspill` process is waiting on the ASR daemon.
    Transcribing,
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            State::Idle => f.write_str("idle"),
            State::Recording { .. } => f.write_str("recording"),
            State::Transcribing => f.write_str("transcribing"),
        }
    }
}

/// Reads the current state, cleaning up any PID file left by a dead process.
pub fn current(paths: &Paths) -> Result<State> {
    if let Some(pid) = live_pid(&paths.recording_pid(), crate::audio::PROCESS_NAME)? {
        return Ok(State::Recording { pid });
    }
    if live_pid(&paths.transcribing_pid(), env!("CARGO_BIN_NAME"))?.is_some() {
        return Ok(State::Transcribing);
    }
    Ok(State::Idle)
}

/// Returns the PID in `path` if that process is still running `comm`.
///
/// Matching on the process name as well as the PID guards against PID reuse:
/// a recycled PID belonging to an unrelated process must never be signalled.
fn live_pid(path: &Path, comm: &str) -> Result<Option<i32>> {
    let Some(pid) = read_pid(path)? else {
        return Ok(None);
    };
    if is_running(pid, comm) {
        return Ok(Some(pid));
    }
    tracing::debug!(pid, file = %path.display(), "removing stale pid file");
    remove_pid(path)?;
    Ok(None)
}

/// True if `pid` exists and its executable name is `comm`.
pub fn is_running(pid: i32, comm: &str) -> bool {
    match fs::read_to_string(format!("/proc/{pid}/comm")) {
        // /proc/<pid>/comm is truncated to 15 characters by the kernel.
        Ok(actual) => actual.trim_end() == &comm[..comm.len().min(15)],
        Err(_) => false,
    }
}

fn read_pid(path: &Path) -> Result<Option<i32>> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("failed to read {}", path.display())),
    };
    match raw.trim().parse::<i32>() {
        Ok(pid) if pid > 0 => Ok(Some(pid)),
        _ => {
            tracing::warn!(file = %path.display(), "discarding malformed pid file");
            remove_pid(path)?;
            Ok(None)
        }
    }
}

pub fn write_pid(path: &Path, pid: i32) -> Result<()> {
    fs::write(path, format!("{pid}\n"))
        .with_context(|| format!("failed to write {}", path.display()))
}

pub fn remove_pid(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("failed to remove {}", path.display())),
    }
}

/// An exclusive `flock` held for the lifetime of a state-changing command.
///
/// Two hotkey presses in quick succession would otherwise race between reading
/// the state and acting on it. The kernel releases the lock when the process
/// exits, so a crash cannot leave it stuck.
pub struct Lock(#[allow(dead_code)] File);

pub fn acquire_lock(paths: &Paths) -> Result<Lock> {
    let path = paths.lock_file();
    let file = File::create(&path)
        .with_context(|| format!("failed to open lock file {}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(Lock(file)),
        Err(TryLockError::WouldBlock) => bail!("another textspill command is already running"),
        Err(TryLockError::Error(e)) => {
            Err(e).with_context(|| format!("failed to lock {}", path.display()))
        }
    }
}

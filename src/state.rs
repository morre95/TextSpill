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
use std::time::{Duration, Instant};

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
    // A failed child can remain in /proc until its parent reaps it. Zombies
    // neither record nor respond to signals and must be treated as stale.
    if let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat"))
        && let Some((_, tail)) = stat.rsplit_once(") ")
        && matches!(tail.as_bytes().first(), Some(b'Z' | b'X'))
    {
        return false;
    }
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

/// How long a command waits for the lock before giving up.
///
/// Commands arrive in quick succession by design — push-to-talk releases the
/// key milliseconds after pressing it, and `stop` must not be refused just
/// because `start` is still confirming that the recorder came up. Waiting
/// serialises them instead; the bound is what keeps a wedged command from
/// blocking the hotkey forever.
const LOCK_TIMEOUT: Duration = Duration::from_secs(3);
const LOCK_POLL: Duration = Duration::from_millis(20);

/// An exclusive `flock` held for the lifetime of a state-changing command.
///
/// Two hotkey presses would otherwise race between reading the state and acting
/// on it. The kernel releases the lock when the process exits, so a crash
/// cannot leave it stuck.
#[derive(Debug)]
pub struct Lock(#[allow(dead_code)] File);

pub fn acquire_lock(paths: &Paths) -> Result<Lock> {
    let path = paths.lock_file();
    let file = File::create(&path)
        .with_context(|| format!("failed to open lock file {}", path.display()))?;

    let deadline = Instant::now() + LOCK_TIMEOUT;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(Lock(file)),
            Err(TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("failed to lock {}", path.display()));
            }
            Err(TryLockError::WouldBlock) if Instant::now() >= deadline => {
                bail!(
                    "another textspill command has held the lock for over {}s",
                    LOCK_TIMEOUT.as_secs()
                );
            }
            Err(TryLockError::WouldBlock) => {
                tracing::debug!("waiting for the runtime lock");
                std::thread::sleep(LOCK_POLL);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A unique directory per test, so tests stay independent under `cargo test`.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("textspill-test-{}-{name}", std::process::id()));
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn own_comm() -> String {
        fs::read_to_string("/proc/self/comm")
            .expect("comm")
            .trim_end()
            .to_string()
    }

    #[test]
    fn is_running_matches_only_the_expected_process() {
        let pid = std::process::id() as i32;
        assert!(is_running(pid, &own_comm()));
        // Same PID, wrong program: this is what a recycled PID looks like.
        assert!(!is_running(pid, "definitely-not-this"));
    }

    #[test]
    fn is_running_is_false_for_a_dead_process() {
        // Above /proc/sys/kernel/pid_max on any realistic system.
        assert!(!is_running(0x3FFF_FFFF, "pw-record"));
    }

    #[test]
    fn pid_file_round_trips() {
        let path = scratch("round-trip").join("recording.pid");
        assert_eq!(read_pid(&path).unwrap(), None, "missing file reads as None");

        write_pid(&path, 4242).unwrap();
        assert_eq!(read_pid(&path).unwrap(), Some(4242));

        remove_pid(&path).unwrap();
        assert_eq!(read_pid(&path).unwrap(), None);
        // Removing again is not an error: cleanup runs on paths that may be gone.
        remove_pid(&path).unwrap();
    }

    #[test]
    fn malformed_pid_file_is_discarded() {
        let path = scratch("malformed").join("recording.pid");
        for junk in ["", "not-a-pid", "0", "-1", "99999999999999999999"] {
            fs::write(&path, junk).unwrap();
            assert_eq!(read_pid(&path).unwrap(), None, "junk: {junk:?}");
            assert!(!path.exists(), "junk {junk:?} should have been removed");
        }
    }

    #[test]
    fn live_pid_keeps_a_matching_process_and_cleans_up_a_stale_one() {
        let path = scratch("live").join("recording.pid");
        let comm = own_comm();

        write_pid(&path, std::process::id() as i32).unwrap();
        assert!(live_pid(&path, &comm).unwrap().is_some());
        assert!(path.exists(), "a live pid file must survive");

        write_pid(&path, 0x3FFF_FFFF).unwrap();
        assert_eq!(live_pid(&path, &comm).unwrap(), None);
        assert!(!path.exists(), "a stale pid file must be removed");
    }

    #[test]
    fn a_waiting_command_gets_the_lock_when_the_holder_finishes() {
        // The push-to-talk case: `stop` arrives while `start` still holds the
        // lock, and must queue behind it rather than be refused.
        let dir = scratch("lock-wait");
        let paths = Paths::for_test(dir);

        let held = acquire_lock(&paths).expect("first holder");
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(120));
            drop(held);
        });

        let started = Instant::now();
        let _second = acquire_lock(&paths).expect("second command must wait, not fail");
        assert!(
            started.elapsed() >= Duration::from_millis(100),
            "it should have waited for the holder"
        );
        handle.join().unwrap();
    }

    #[test]
    fn waiting_for_the_lock_gives_up_eventually() {
        let dir = scratch("lock-timeout");
        let paths = Paths::for_test(dir);
        let _held = acquire_lock(&paths).expect("first holder");

        let started = Instant::now();
        let e = acquire_lock(&paths).expect_err("a wedged holder must not block forever");
        assert!(e.to_string().contains("held the lock"), "{e}");
        assert!(started.elapsed() >= LOCK_TIMEOUT);
        assert!(started.elapsed() < LOCK_TIMEOUT + Duration::from_secs(2));
    }

    #[test]
    fn the_lock_is_exclusive_and_released_on_drop() {
        let dir = scratch("lock");
        let path = dir.join("textspill.lock");

        let first = File::create(&path).unwrap();
        first.try_lock().expect("first holder acquires");
        let second = File::create(&path).unwrap();
        assert!(
            matches!(second.try_lock(), Err(TryLockError::WouldBlock)),
            "a second command must not acquire the lock"
        );

        drop(first);
        File::create(&path)
            .unwrap()
            .try_lock()
            .expect("lock is released when the holder exits");
    }

    #[test]
    fn state_is_displayed_as_the_documented_status_words() {
        assert_eq!(State::Idle.to_string(), "idle");
        assert_eq!(State::Recording { pid: 1 }.to_string(), "recording");
        assert_eq!(State::Transcribing.to_string(), "transcribing");
    }
}

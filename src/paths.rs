//! Every file TextSpill touches is named here, so that nothing else in the
//! codebase hardcodes a path. Adding `$XDG_CONFIG_HOME/textspill/config.toml`
//! later means adding one accessor, not hunting through modules.

use anyhow::{Context, Result, bail};
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::PathBuf;

/// Runtime state is per-user and never world-readable.
const RUNTIME_MODE: u32 = 0o700;

/// Resolved locations for one invocation of `textspill`.
pub struct Paths {
    runtime_dir: PathBuf,
}

impl Paths {
    pub fn config_dir() -> Result<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")))
            .context("HOME or XDG_CONFIG_HOME is required for TextSpill settings")?;
        Ok(base.join("textspill"))
    }

    pub fn session_config(&self) -> PathBuf {
        self.runtime_dir.join("session-config.json")
    }
    pub fn stream_state(&self) -> PathBuf {
        self.runtime_dir.join("stream.json")
    }
    /// Resolves `$XDG_RUNTIME_DIR/textspill/`, creating it with mode 0700.
    ///
    /// Refuses to use a directory owned by anyone else: on the `/tmp` fallback
    /// path that would let another user plant a socket or a PID file.
    pub fn new() -> Result<Self> {
        let runtime_dir = runtime_base().join("textspill");
        fs::create_dir_all(&runtime_dir)
            .with_context(|| format!("failed to create runtime dir {}", runtime_dir.display()))?;

        let meta = fs::metadata(&runtime_dir)?;
        let uid = nix::unistd::getuid().as_raw();
        if meta.uid() != uid {
            bail!(
                "runtime dir {} is owned by uid {}, not {uid}",
                runtime_dir.display(),
                meta.uid()
            );
        }
        if meta.permissions().mode() & 0o777 != RUNTIME_MODE {
            fs::set_permissions(&runtime_dir, fs::Permissions::from_mode(RUNTIME_MODE))?;
        }

        tracing::debug!(runtime_dir = %runtime_dir.display(), "resolved runtime dir");
        Ok(Self { runtime_dir })
    }

    /// PID of the running `pw-record`, present only while recording.
    pub fn recording_pid(&self) -> PathBuf {
        self.runtime_dir.join("recording.pid")
    }

    /// PID of the `textspill` process currently talking to the ASR daemon.
    pub fn transcribing_pid(&self) -> PathBuf {
        self.runtime_dir.join("transcribing.pid")
    }

    /// The captured audio. Kept on failure so a dictation is never lost.
    pub fn recording_wav(&self) -> PathBuf {
        self.runtime_dir.join("recording.wav")
    }

    /// `pw-record`'s stderr, so a failed capture can be explained.
    pub fn capture_log(&self) -> PathBuf {
        self.runtime_dir.join("pw-record.log")
    }

    /// Where `textspill-asr.service` listens.
    pub fn asr_socket(&self) -> PathBuf {
        self.runtime_dir.join("asr.sock")
    }

    /// Serialises concurrent `textspill` invocations (double hotkey presses).
    pub fn lock_file(&self) -> PathBuf {
        self.runtime_dir.join("textspill.lock")
    }

    pub fn live_session(&self) -> PathBuf {
        self.runtime_dir.join("live-session")
    }

    pub fn preview(&self) -> PathBuf {
        self.runtime_dir.join("preview.json")
    }

    pub fn live_log(&self) -> PathBuf {
        self.runtime_dir.join("live.log")
    }

    pub fn live_audio(&self, session: &str) -> PathBuf {
        self.runtime_dir.join(format!("preview-{session}.wav"))
    }
}

#[cfg(test)]
impl Paths {
    /// Builds `Paths` over an existing directory, for tests that must not touch
    /// the real runtime dir.
    pub fn for_test(runtime_dir: PathBuf) -> Self {
        Self { runtime_dir }
    }
}

/// `$XDG_RUNTIME_DIR`, or a private `/tmp` directory when it is unset.
fn runtime_base() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return dir;
        }
        tracing::warn!(dir = %dir.display(), "XDG_RUNTIME_DIR is not a directory");
    }
    let uid = nix::unistd::getuid().as_raw();
    let fallback = PathBuf::from(format!("/run/user/{uid}"));
    if fallback.is_dir() {
        return fallback;
    }
    PathBuf::from(format!("/tmp/textspill-{uid}"))
}

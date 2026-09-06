//! Wayland clipboard, via `wl-copy`.
//!
//! Text is handed over on stdin, never as an argument: transcriptions are
//! user-controlled data and must not become part of a command line.

use anyhow::{Context, Result, bail};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const COPY_TOOL: &str = "wl-copy";
const PASTE_TOOL: &str = "wl-paste";

/// How long the compositor gets to publish the new selection.
const SETTLE_TIMEOUT: Duration = Duration::from_millis(500);
const SETTLE_POLL: Duration = Duration::from_millis(15);

/// Puts `text` on the clipboard and waits until it can be read back.
///
/// `wl-copy` forks into the background before the selection is actually offered
/// to the compositor, so its exit status alone does not mean the clipboard is
/// ready. Polling `wl-paste` turns a guessed sleep into a measured fact, which
/// is what keeps the subsequent paste keystroke from landing too early.
pub fn copy(text: &str) -> Result<()> {
    // `--trim-newline` plus already-sanitised input: a trailing newline in the
    // clipboard would press Enter when pasted into a shell.
    let mut child = Command::new(COPY_TOOL)
        .arg("--trim-newline")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to run {COPY_TOOL} (is wl-clipboard installed?)"))?;

    child
        .stdin
        .take()
        .context("wl-copy stdin was not available")?
        .write_all(text.as_bytes())
        .context("failed to write transcription to wl-copy")?;

    let status = child.wait().context("failed to wait for wl-copy")?;
    if !status.success() {
        bail!("{COPY_TOOL} exited with {status}");
    }

    wait_until_readable(text)?;
    tracing::info!(chars = text.chars().count(), "copied to clipboard");
    Ok(())
}

fn wait_until_readable(expected: &str) -> Result<()> {
    let deadline = Instant::now() + SETTLE_TIMEOUT;
    loop {
        if read_back().as_deref() == Some(expected) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("clipboard did not contain the transcription after copying");
        }
        std::thread::sleep(SETTLE_POLL);
    }
}

fn read_back() -> Option<String> {
    let output = Command::new(PASTE_TOOL)
        .arg("--no-newline")
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
}

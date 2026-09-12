//! Persistent settings contain no credentials. A session freezes non-secret settings.
use crate::paths::Paths;
use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Local,
    Deepgram,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub backend: Backend,
    pub deepgram_language: String,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::Local,
            deepgram_language: "sv".into(),
            extra: BTreeMap::new(),
        }
    }
}

impl Config {
    pub fn read() -> Result<Self> {
        Self::read_at(&Paths::config_dir()?.join("config.json"))
    }

    fn read_at(path: &Path) -> Result<Self> {
        match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("invalid TextSpill config.json"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).context("cannot read TextSpill config.json"),
        }
    }

    pub fn effective() -> Result<Self> {
        let mut config = Self::read()?;
        if let Ok(value) = std::env::var("TEXTSPILL_BACKEND") {
            config.backend = match value.as_str() {
                "local" => Backend::Local,
                "deepgram" => Backend::Deepgram,
                _ => bail!("TEXTSPILL_BACKEND must be local or deepgram"),
            };
        }
        if let Ok(value) = std::env::var("TEXTSPILL_DEEPGRAM_LANGUAGE") {
            config.deepgram_language = value;
        }
        if config.backend == Backend::Deepgram {
            config.validate_language()?;
            api_key()?;
        }
        Ok(config)
    }

    pub fn validate_language(&self) -> Result<()> {
        if !matches!(self.deepgram_language.as_str(), "sv" | "en") {
            bail!("deepgram_language must be sv or en");
        }
        Ok(())
    }
}

pub fn configure(backend: Option<Backend>) -> Result<()> {
    let mut config = Config::read()?;
    if let Some(backend) = backend {
        config.backend = backend;
        let dir = Paths::config_dir()?;
        fs::create_dir_all(&dir)?;
        write_json(&dir.join("config.json"), &config)?;
    }
    println!(
        "{}",
        match config.backend {
            Backend::Local => "local",
            Backend::Deepgram => "deepgram",
        }
    );
    Ok(())
}

pub fn api_key() -> Result<String> {
    let value = match std::env::var("DEEPGRAM_API_KEY") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => {
            let path = Paths::config_dir()?.join("deepgram-api-key");
            fs::read_to_string(path).context(
                "set DEEPGRAM_API_KEY or create ~/.config/textspill/deepgram-api-key (mode 0600)",
            )?
        }
        Err(_) => bail!("DEEPGRAM_API_KEY is not valid text"),
    };
    let key = value.trim();
    if key.is_empty() || !key.bytes().all(|b| b.is_ascii_graphic()) {
        bail!("Deepgram API key is empty or invalid; configure it before recording");
    }
    Ok(key.to_owned())
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub config: Config,
    pub live: bool,
}

impl SessionConfig {
    pub fn load(paths: &Paths) -> Result<Self> {
        // Recordings from an older binary are necessarily local.
        match fs::read(paths.session_config()) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("invalid recording settings"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self {
                config: Config::default(),
                live: false,
            }),
            Err(e) => Err(e.into()),
        }
    }
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    use std::{io::Write, os::unix::fs::OpenOptionsExt};
    let temp = path.with_extension("tmp");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&temp)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    fs::rename(temp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_installations_default_to_local_and_settings_round_trip() {
        let path = std::env::temp_dir().join(format!("ts-config-{}.json", std::process::id()));
        assert_eq!(Config::read_at(&path).unwrap().backend, Backend::Local);
        let mut config: Config =
            serde_json::from_str(r#"{"backend":"deepgram","custom":42}"#).unwrap();
        assert_eq!(config.deepgram_language, "sv");
        config.deepgram_language = "en".into();
        write_json(&path, &config).unwrap();
        let stored = Config::read_at(&path).unwrap();
        assert_eq!(stored.extra["custom"], 42);
        assert_eq!(stored.deepgram_language, "en");
        fs::remove_file(path).unwrap();
    }
}

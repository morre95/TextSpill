//! Backend-neutral final transcription. Local preview remains segmented inference.
use crate::{
    config::{Backend, Config},
    deepgram, ipc,
    paths::Paths,
};
use anyhow::Result;
use std::{path::Path, time::Duration};

#[derive(Debug)]
pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
}

pub fn transcribe(paths: &Paths, config: &Config, wav: &Path) -> Result<Transcription> {
    match config.backend {
        Backend::Local => ipc::transcribe(&paths.asr_socket(), wav, Duration::from_secs(120)),
        Backend::Deepgram => deepgram::transcribe(config, wav),
    }
}

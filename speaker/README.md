# CAM++ / WeSep feasibility evaluation

This implements the **model-evaluation stage** of the speaker-filter plan.
It is an offline developer tool, not an enabled TextSpill feature. The Rust
client, installation, ASR backends and paste behavior are unchanged.

**Integration gate: not passed.** CPU throughput looks feasible, but extraction
quality and a safe verification threshold are not established for the requested
multi-speaker environment. See [the measured results](EVALUATION.md).

## Reproduce

Requires Python 3.12, Git and internet for setup. All downloaded source, model
weights, public example audio and the virtualenv live under `target/speaker-eval`.
No microphone, GPU, API key or running TextSpill service is used.

```bash
bash speaker/setup_eval.sh
bash speaker/run_eval.sh --threads 4
target/speaker-eval/venv/bin/python -m unittest discover -s speaker -p 'test_*.py' -v
```

The runner prints each measured window and saves `target/speaker-eval/report.json`.
Model loading and one warm-up pass are excluded from the timing samples. Each
window is measured three times by default. Real-time factor (RTF) is processing
seconds divided by audio seconds; less than 1 means faster than real time.
The timing includes separation, CAM++ feature computation and verification,
but excludes ASR and most file I/O. Overlapping live windows would add work.

Exit code 0 means every window's **median RTF** is below 1, not that the feature
passed its quality gate. Exit code 2 means the CPU timing criterion failed.
Invalid audio, models or manifests fail with a nonzero exit code. Reports always
state `production_ready: false` and do not select an acceptance threshold.

## Models and preprocessing

- Verification: WeSpeaker VoxCeleb `voxceleb_CAM++.onnx` (not the LM variant),
  restricted to ONNX Runtime's CPU provider. Embeddings use the upstream
  `infer_onnx.py` preprocessing: mono 16 kHz float samples scaled by 32768,
  80-bin Kaldi fbank, 25 ms frames / 10 ms shift, Hamming window, no dither or
  energy feature, and mean subtraction across frames. Normalize embeddings
  before cosine comparison; scores are not probabilities.
- Separation: WeSep `bsrnn_ecapa_vox1`, using its original ECAPA voice encoder
  and reference audio. CAM++ embeddings are **not** substituted for the
  checkpoint's internal speaker representation. Output peak normalization is
  disabled to avoid amplifying residual speech when the target is absent.
- Both weights and WeSep's config are verified by SHA-256 before loading.
  The setup script pins source revisions and Python dependencies.
- WeSep imports Silero VAD, which resets PyTorch's thread count to 1 at import
  time. The evaluator restores the requested thread count after loading models;
  a regression test covers this behavior.

Sources: [WeSpeaker models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md),
[CAM++ preprocessing](https://github.com/wenet-e2e/wespeaker/blob/820acb41d3ea1ffe2c465375189d7578f4553996/wespeaker/bin/infer_onnx.py),
[WeSep extractor](https://github.com/wenet-e2e/wesep/blob/99eca54b60300d39b9353d93cf285a14bba37854/wesep/cli/extractor.py),
[public demo](https://huggingface.co/spaces/wenet-e2e/wesep-tse-2speaker-demo/tree/c3212546b3d42328c059562b7a94508dd3833094).

## Evaluation audio

The default fixtures are generated from the upstream demo's public recordings.
Enrollment uses the first 3 seconds of `enroll_1.wav`; clean target test audio
uses seconds 3–5 with no overlapping samples. This same-recording split is an
optimistic smoke test, **not** independent-session verification calibration.
Cases include target alone, other alone, two speakers, a louder interferer,
silence, the official mixture, three speakers, two nontarget speakers and a
speaker change. The third speaker is from a Chinese clip, resampled to 16 kHz.
The three-speaker mixture normalizes the speakers to equal RMS before mixing.

Constructed mixtures have sample-aligned clean targets for SI-SDR. Higher SI-SDR
means less reconstruction error relative to the target; it does **not** measure
which words ASR will recognize. The official mixture has no isolated reference,
so only timing and speaker similarity are reported for it.

To evaluate additional recordings, supply a manifest; relative paths are
resolved against its directory:

```json
{
  "enrollment": "reference.wav",
  "limitations": ["Describe the source and recording conditions here"],
  "cases": [
    {
      "name": "swedish-overlap",
      "audio": "mixture.wav",
      "clean_target": "isolated-target.wav",
      "target_present": true,
      "language": "sv"
    },
    {
      "name": "other-speaker-only",
      "audio": "other.wav",
      "target_present": false,
      "language": "sv"
    }
  ]
}
```

`clean_target` is optional; when present it must match the mixture length.
Provide 15–30 seconds of clean enrollment for the intended product scenario.
Use independently recorded positive/negative examples for calibration and a
separate held-out test set. Files must be mono 16 kHz. No automatic resampling
of user-supplied evaluation audio occurs.

```bash
bash speaker/run_eval.sh --threads 4 \
  --manifest /absolute/path/to/manifest.json \
  --output target/speaker-eval/custom-report.json
```

## Remaining product work

Before integration, validate Swedish and English with realistic enrollment,
independent recordings, multiple microphones, overlap, noise and short speech.
Calibrate rejection thresholds and measure both false acceptance and loss of
the target's speech. Evaluate the words produced by both ASR backends, since
speaker similarity alone cannot certify clean target-only audio.

After that gate passes, implement the profile CLI, private warm speaker service,
overlapping-window processing, session isolation, filtered audio feeding both
ASR backends, preview diagnostics and installer support. None of those product
capabilities is advertised or enabled by this evaluation-only change.

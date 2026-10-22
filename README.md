# VoiceRail

Train your own voice models from a handful of recordings and give any song or
speech a new voice, all from a local web UI.

VoiceRail is built on RVC (Retrieval-based Voice Conversion) v2, the format
used by RVC WebUI, Applio and Ultimate RVC. Models trained here work in those
tools, and community models work here.

## What it does

**Convert.** Upload a song or a voice recording, pick a voice model, press
*Convert*. For songs, VoiceRail:

1. separates the vocals from the music (Mel-Band RoFormer, the current state
   of the art for vocal isolation),
2. optionally splits lead and backing vocals so only the lead is converted,
3. removes reverb/echo so the voice converts cleanly,
4. converts the voice (RMVPE pitch tracking, retrieval index, consonant
   protection, optional auto pitch matching),
5. adds a touch of reverb back and remixes everything with the music.

You get the final mix plus every stem (converted vocals, music, original
vocals, backing vocals) as WAV, FLAC or MP3.

**Train.** Upload recordings of a voice (mp3, wav, flac, m4a, even videos),
give it a name and press *Start training*. VoiceRail:

1. cleans every file: removes music and background, backing vocals and
   harmonies, reverb/echo and noise (each step optional, with presets for
   songs, clean speech and noisy recordings),
2. normalises loudness, removes rumble, slices at silences and drops unusable
   segments (too short, silent, unvoiced),
3. extracts pitch and voice features,
4. fine-tunes a pretrained RVC v2 model, scoring held-out clips through the
   real conversion path to keep the **best** epoch and to stop automatically
   when the voice stops improving,
5. builds the retrieval index and adds the finished model to your library.

The Train tab shows live progress, a loss chart, a preview clip after each
save (next to the original), the cleaned training audio and a dataset report
with warnings (for example "only 4 minutes of clean voice").

**Voice models.** Import `.pth`/`.index` files, `.zip` archives or download
links (Hugging Face etc.), rename, delete and export models.

## Improvements over typical RVC tools

- **Training data cleaning:** vocal isolation, backing vocal removal,
  de-reverb and denoise with RoFormer models before training. Plain RVC
  trains on whatever you feed it.
- **Better epoch selection:** a validation set scored through the inference
  path picks the best checkpoint, instead of guessing from the GAN loss.
- **Stable training:** single process (no `torch.distributed`, so no port or
  socket errors), automatic batch size and epochs, bf16 on modern GPUs,
  NaN/inf steps skipped instead of corrupting the model, graceful stop with
  resume at any time.
- **Training survives the browser:** it runs in a background process. Close
  the tab, come back later and the UI picks it up again.
- **Cleaner pitch:** octave glitches, micro-dropouts and single-frame jitter
  are fixed before conversion and training.
- **Exact timing:** long inputs are converted in chunks without the small
  timing drift classic RVC accumulates at chunk boundaries, so vocals stay
  locked to the music.
- **Safe model loading:** model files are opened with PyTorch's
  `weights_only` loader, so a malicious `.pth` cannot run code. Archives are
  extracted safely.
- **Reproducible install:** every dependency is pinned in `uv.lock`;
  checksummed, resumable downloads for base models.

## Requirements

- Windows 10/11 or Linux (macOS on Apple Silicon works on the CPU).
- An NVIDIA GPU is strongly recommended: 4 GB VRAM for converting, 6 GB+ for
  training. AMD GPUs work on Linux with ROCm. Without a GPU, conversion works
  (slowly) and training is impractical.
- About 10 GB of free disk space for models and caches.
- Internet on first use (to download the base and separation models).

## Install and run

Each platform has its own launcher; none of them needs the other's tools.

**Windows (development machine):** double-click `voicerail.bat`, or run it
from a terminal with any `voicerail` command, e.g. `voicerail.bat models`.

**Linux / macOS:**

```bash
./voicerail.sh
```

**Linux GPU pods:** `./pod.sh` or the notebook, see
[Remote GPU pods](#remote-gpu-pods-runpod-vastai-lambda-) below.

The first run installs [uv](https://docs.astral.sh/uv/), Python 3.12 and all
dependencies (a few GB, takes a while), then opens the web UI at
<http://127.0.0.1:8080>. Later runs start in seconds.

The launcher picks the right PyTorch build from your GPU driver:

| Variant | When | Override |
|---|---|---|
| `cuda` | NVIDIA driver 580 or newer (all GPUs, including RTX 50xx) | `VOICERAIL_TORCH=cuda` |
| `cuda-legacy` | NVIDIA driver 525–579 | `VOICERAIL_TORCH=cuda-legacy` |
| `rocm` | AMD GPU on Linux | `VOICERAIL_TORCH=rocm` |
| `cpu` | everything else | `VOICERAIL_TORCH=cpu` |

Manual installation with uv:

```bash
uv sync --extra cuda      # or cuda-legacy / rocm / cpu
uv run --no-sync voicerail
```

All data (models, training sessions, outputs, caches) goes to `./data`. Set
`VOICERAIL_HOME` to use another folder.

VoiceRail always uses the GPU when PyTorch can run on it and falls back to
the CPU otherwise (with a warning in the log and a badge in the UI). Set
`VOICERAIL_DEVICE=cpu` (or `cuda:1`) to force a device.

The web server can also be configured through the environment, which is
handy on headless machines: `VOICERAIL_HOST` (default `127.0.0.1`; use
`0.0.0.0` to accept connections from other machines), `VOICERAIL_PORT`,
`VOICERAIL_SHARE=1` (public link) and `VOICERAIL_NO_BROWSER=1`.

## Remote GPU pods (RunPod, Vast.ai, Lambda, ...)

Rent a pod with an NVIDIA GPU and Jupyter (any PyTorch/CUDA template works,
no Docker image of your own is needed), then either:

- **Notebook:** upload [`VoiceRail.ipynb`](VoiceRail.ipynb) to the pod's
  Jupyter, set `REPO_URL` in the first cell and run the cells. It clones the
  repository, installs everything, downloads the base models and starts the
  web UI with a public link.
- **Terminal:** in a Jupyter terminal:

  ```bash
  git clone <your repo URL> /workspace/VoiceRail && cd /workspace/VoiceRail
  ./pod.sh
  ```

`pod.sh` keeps Python, packages, base models, training sessions and outputs
on the persistent volume (`/workspace` when present, `VOICERAIL_VOLUME` to
choose another; plan on 20 GB, the Linux CUDA PyTorch stack alone is about
8 GB), binds the UI to `0.0.0.0:8080` and prints a public
`*.gradio.live` link. Set `VOICERAIL_SHARE=0` to use the pod's own port proxy
instead (expose HTTP port 8080 in the pod template). Any `voicerail` command
works through it, e.g. `./pod.sh train MyVoice --data /workspace/recordings`
or `./pod.sh download` to prepare a pod ahead of time. Training keeps running
in its own process if the notebook or the browser tab is closed.

## Training tips

- **Amount:** 10–30 minutes of a single voice is ideal; 5 minutes works,
  more than an hour rarely helps.
- **Quality beats quantity:** solo vocals, podcasts and audiobooks are best.
  Songs work thanks to the cleaning steps, but heavy effects (autotune,
  distortion, doubled vocals) end up in the model.
- **Variety:** include the range you want to reproduce: low and high notes,
  soft and loud, different vowels.
- **Only one speaker:** remove interviews, duets and featured artists.
- **Sample rate:** 40 kHz is the default and best-supported; use 48 kHz for
  hi-fi singing.
- Leave epochs and batch size on automatic and keep *Stop when it stops
  improving* on. Listen to the preview clips: when they sound like the target
  voice, you can stop and press *Save current model now*.
- The defaults are the best general settings. *Training settings* (learning
  rate and decay, loss weights, precision, patience, low-VRAM mode) and
  *Dataset slicing* (segment length, silence threshold, loudness) are there
  for experiments; every control says what raising or lowering it does.

## Conversion tips

- **Pitch shift:** +12 converts a male voice to a female range, -12 the other
  way. *Auto-match* computes it from the model's training data (models
  trained with VoiceRail), using octave steps for songs to stay in key.
- **Timbre retrieval (index rate):** higher sounds more like the target, lower
  gives clearer pronunciation. 0.5–0.75 works well.
- **Consonant protection:** lower it (0.2–0.3) if breaths or "s" sounds
  become metallic.
- Keep *Remove reverb* on for songs; dry vocals convert far better.
- *Conversion settings* also has the volume envelope, autotune and the pitch
  detection threshold (raise it if breathy passages produce stray notes).

## Command line

```bash
voicerail                                   # web UI
voicerail convert song.mp3 -m MyVoice -p 12 # convert a song
voicerail convert talk.wav -m MyVoice --voice --auto-pitch -f flac
voicerail train MyVoice --data recordings/ --preset songs
voicerail import https://huggingface.co/user/repo/resolve/main/model.zip
voicerail models                            # list models
voicerail download                          # fetch base models up front
```

Use `uv run --no-sync voicerail ...` when the virtual environment is not
activated.

## Troubleshooting

- **CUDA out of memory while training:** lower the batch size or enable
  *Low VRAM mode* in *Training settings*.
- **Training is unstable / NaN errors:** set *Precision* to `fp32`.
- **Separation is very slow:** you are probably running on the CPU. Install
  the CUDA build, or choose *MDX-Net Voc FT (fast)*.
- **"No usable voice was found":** the file has no singing/speech after
  cleaning. Try the *Clean voice* preset if your recordings have no music.
- Detailed logs of every training session are in
  `data/training/<name>/log.txt`.

## Development

```bash
uv sync --extra cuda-legacy --group dev     # or cuda / rocm / cpu, see above
uv run --no-sync pytest                     # unit tests
VOICERAIL_TEST_ASSETS=data uv run --no-sync pytest -m assets   # end-to-end
uv run --no-sync ruff check src tests && uv run --no-sync ruff format src tests
```

On Windows set the variable first: `$env:VOICERAIL_TEST_ASSETS = "data"`
(PowerShell). The end-to-end test trains a tiny model and converts a clip
through the real pipeline, on the GPU when one is available.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Only use voices you have the right to use.

# Third-party notices

VoxShift is GPL-3.0 (see `LICENSE`). The Windows build bundles the
components below. Each remains under its own license; full texts are at the
linked projects.

## AI models (exported to ONNX and shipped in `models/onnx/whisper-small-bigvgan/`)

| component | source | license |
|---|---|---|
| Seed-VC DiT + length regulator (`DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned`) | [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC), code [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | GPL-3.0 |
| Whisper-small encoder | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | Apache-2.0 |
| BigVGAN v2 22 kHz 80-band vocoder | [nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) | MIT, © NVIDIA |
| CAMPPlus speaker embedding | [funasr/campplus](https://huggingface.co/funasr/campplus) | Apache-2.0 |

The ONNX files are produced from these checkpoints by `voxshift export-onnx`
(`voxshift/ai/onnx_export.py`). The Seed-VC source used to build and export
them is pinned as a git submodule in `third_party/seed-vc`.

## Runtime libraries

| component | license |
|---|---|
| Python 3.12 | PSF License |
| NumPy, SciPy | BSD-3-Clause |
| ONNX Runtime (DirectML build) | MIT, © Microsoft |
| DirectML (`DirectML.dll`, shipped with ONNX Runtime) | Microsoft DirectML redistributable license |
| sounddevice / PortAudio | MIT |
| soundfile / libsndfile | BSD-3-Clause / LGPL-2.1 (dynamically linked) |
| Pillow | MIT-CMU |
| pystray | LGPL-3.0 (dynamically loaded Python module) |
| sv-ttk (Sun Valley theme) | MIT |
| Tcl/Tk | Tcl/Tk License |
| OpenSSL, libffi (bundled with Python) | Apache-2.0, MIT |
| PyInstaller bootloader | GPL-2.0 with bootloader exception |

## Not bundled

**VB-CABLE** (virtual audio cable) by VB-Audio Software is donationware. The
installer downloads it from vb-audio.com at install time and runs it only
after verifying VB-Audio's Authenticode signature. It is not redistributed
with VoxShift.

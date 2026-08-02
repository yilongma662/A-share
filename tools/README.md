# upscale.py

Tiled [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) upscaler that runs on CUDA, Apple MPS or CPU.

## Install

CPU only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchvision --index-url https://download.pytorch.org/whl/cpu
pip install spandrel pillow numpy
```

NVIDIA GPU (pick the CUDA build matching your driver):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install spandrel pillow numpy
```

Install `torch` and `torchvision` from the same index. Mixing a CPU `torch` with a
PyPI `torchvision` fails at import with `operator torchvision::nms does not exist`.

## Weights

```bash
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
```

`RealESRGAN_x4plus` keeps painterly brush texture. `RealESRGAN_x4plus_anime_6B` is
4x smaller and faster, and gives cleaner line art at the cost of flattening texture.

## Usage

```bash
python tools/upscale.py in.png out.png --weights RealESRGAN_x4plus.pth --half
```

A second pass over an already upscaled image mostly invents smooth detail, so
`--down 2` after 4x weights yields a crisper 2x than magnifying straight to 4x:

```bash
python tools/upscale.py 4x.png 8x.jpg --weights RealESRGAN_x4plus.pth --half --down 2
```

Lower `--tile` if you run out of memory; raise it to cut per-tile overhead.

## Speed

Measured on this repository's artwork with `RealESRGAN_x4plus`, 4 CPU cores, no GPU:

| Input | Output | Time |
| --- | --- | --- |
| 1536×1024 | 6144×4096 | 173 s |
| 6144×4096 | 12288×8192 (4x then `--down 2`) | ~28 min |

That is roughly 67 s per megapixel of input. A mid-range discrete GPU typically runs
the same model at well under 2 s per megapixel, so the same jobs finish in seconds.

`--half` uses bfloat16 on CPU for a 1.8x speedup; measured against fp32 the mean
absolute pixel difference was 0.66/255, which is not visible.

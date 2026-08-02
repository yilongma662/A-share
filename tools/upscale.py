"""Tiled Real-ESRGAN upscaler.

Runs on CUDA, Apple MPS or CPU, picking whichever is available. Large images are
processed tile by tile so peak memory stays close to a single output buffer
rather than scaling with the full result.

    python tools/upscale.py in.png out.png --weights RealESRGAN_x4plus.pth

Weights (any Real-ESRGAN / spandrel-supported checkpoint works):
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

Image.MAX_IMAGE_PIXELS = None


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--weights", required=True)
    p.add_argument("--tile", type=int, default=512, help="tile size in source pixels")
    p.add_argument("--pad", type=int, default=32, help="context pixels fed around each tile")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--half", action="store_true", help="fp16 on CUDA, bfloat16 on CPU")
    p.add_argument(
        "--down",
        type=int,
        default=1,
        help="downsample the result by this factor, e.g. 4x weights with --down 2 for a crisp 2x",
    )
    p.add_argument("--jpeg-quality", type=int, default=95)
    args = p.parse_args()

    device = pick_device(args.device)
    torch.set_grad_enabled(False)

    model = ModelLoader().load_from_file(args.weights)
    assert isinstance(model, ImageModelDescriptor)
    model.eval().to(device)
    scale = model.scale

    if args.half and device.type == "cuda":
        model.model.half()

    img = Image.open(args.src).convert("RGB")
    w, h = img.size
    src = np.asarray(img, dtype=np.uint8)
    del img

    print(f"{model.architecture.name} scale={scale} device={device} src={w}x{h}", flush=True)
    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    ys = range(0, h, args.tile)
    xs = range(0, w, args.tile)
    total = len(ys) * len(xs)
    start = time.time()
    done = 0

    for y in ys:
        for x in xs:
            y1, x1 = min(h, y + args.tile), min(w, x + args.tile)
            py0, py1 = max(0, y - args.pad), min(h, y1 + args.pad)
            px0, px1 = max(0, x - args.pad), min(w, x1 + args.pad)

            patch = torch.from_numpy(src[py0:py1, px0:px1].copy()).permute(2, 0, 1)
            patch = patch.unsqueeze(0).to(device).float().div_(255.0)

            if args.half and device.type == "cuda":
                res = model(patch.half()).float()
            elif args.half and device.type == "cpu":
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    res = model(patch)
                res = res.float()
            else:
                res = model(patch)
            res = res.clamp(0, 1).mul(255.0).round().to(torch.uint8).cpu()

            top, left = (y - py0) * scale, (x - px0) * scale
            core = res[0, :, top : top + (y1 - y) * scale, left : left + (x1 - x) * scale]
            out[y * scale : y1 * scale, x * scale : x1 * scale] = core.permute(1, 2, 0).numpy()

            done += 1
            el = time.time() - start
            print(f"tile {done}/{total} elapsed={el:.0f}s eta={el / done * (total - done):.0f}s", flush=True)

    result = Image.fromarray(out)
    if args.down > 1:
        result = result.resize((w * scale // args.down, h * scale // args.down), Image.LANCZOS)

    if args.dst.lower().endswith((".jpg", ".jpeg")):
        result.save(args.dst, quality=args.jpeg_quality, subsampling=0)
    else:
        result.save(args.dst)
    print(f"saved {args.dst} {result.width}x{result.height} in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

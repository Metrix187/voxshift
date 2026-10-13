"""Render the VoxShift icon (assets/voxshift.ico + .png). Run once; outputs are committed."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "voxshift" / "assets"
S = 512


def render() -> Image.Image:
    grad = Image.new("RGBA", (S, S))
    px = grad.load()
    a, b = (124, 58, 237), (6, 182, 212)  # violet -> cyan
    for y in range(S):
        for x in range(S):
            t = (x + y) / (2 * S - 2)
            px[x, y] = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((16, 16, S - 16, S - 16), radius=112, fill=255)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    heights = [0.22, 0.42, 0.70, 0.95, 0.62, 0.85, 0.48, 0.28]
    w, gap = 34, 18
    x0 = (S - (len(heights) * w + (len(heights) - 1) * gap)) // 2
    for i, h in enumerate(heights):
        bh = int(h * 300)
        x = x0 + i * (w + gap)
        d.rounded_rectangle((x, S // 2 - bh // 2, x + w, S // 2 + bh // 2), radius=w // 2, fill="white")
    return img


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    img = render()
    img.resize((256, 256), Image.LANCZOS).save(OUT / "voxshift.png")
    img.save(OUT / "voxshift.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", OUT / "voxshift.ico")

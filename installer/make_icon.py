"""Рисует installer/winai.ico (значок exe и установщика)."""
from pathlib import Path

from PIL import Image, ImageDraw

img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((8, 8, 248, 248), 56, fill=(0, 103, 192))
d.text((128, 132), "AI", fill="white", font_size=120, anchor="mm")
img.save(Path(__file__).with_name("winai.ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

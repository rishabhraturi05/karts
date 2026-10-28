import cv2
import numpy as np
from pathlib import Path

P = Path(__file__).resolve().parent
mask = cv2.imread(str(P / "track_mask.png"), 0)
if mask is None:
    raise FileNotFoundError("track_mask.png")
y, x = np.where(mask == 255)
pixels = np.column_stack((x, y))
np.save(P / "track_pixels.npy", pixels)
print("Valid pixels:", len(pixels))

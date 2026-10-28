import cv2
import numpy as np
from pathlib import Path

P = Path(__file__).resolve().parent
image = cv2.imread(str(P / "track_reference.png"))
if image is None:
    raise FileNotFoundError("track_reference.png")
mask = np.zeros(image.shape[:2], np.uint8)
brush = 15
mode = None


def mouse(event, x, y, flags, data):
    global mode
    if event == cv2.EVENT_LBUTTONDOWN:
        mode = 255
    if event == cv2.EVENT_RBUTTONDOWN:
        mode = 0
    if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        mode = None
    if mode is not None and (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        cv2.circle(mask, (x, y), brush, mode, -1)


cv2.namedWindow("Paint mask")
cv2.setMouseCallback("Paint mask", mouse)

while True:
    overlay = image.copy()
    overlay[mask == 255] = (0, 255, 0)
    cv2.imshow("Paint mask", cv2.addWeighted(image, 0.6, overlay, 0.4, 0))
    key = cv2.waitKey(20) & 0xFF
    if key == ord("s"):
        cv2.imwrite(str(P / "track_mask.png"), mask)
        print("Mask saved")
    elif key == ord("r"):
        mask[:] = 0
    elif key in (ord("+"), ord("=")):
        brush += 3
    elif key == ord("-"):
        brush = max(2, brush - 3)
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

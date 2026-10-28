import cv2
import numpy as np
from pathlib import Path

P = Path(__file__).resolve().parent
img = cv2.imread(str(P / "track_reference.png"))
mask = cv2.imread(str(P / "track_mask.png"), 0)
line = np.load(P / "dense_centerline.npy")
cum = np.load(P / "cumulative_distance.npy")
length = float(np.load(P / "lap_length.npy"))


def mouse(event, x, y, flags, data):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if mask[y, x] != 255:
        print((x, y), "OFF TRACK")
        return
    idx = np.argmin(np.sum((line - np.array([x, y])) ** 2, axis=1))
    print((x, y), "ON TRACK", f"{cum[idx] / length * 100:.2f}%")


cv2.namedWindow("Test progress")
cv2.setMouseCallback("Test progress", mouse)

while True:
    view = img.copy()
    cv2.polylines(view, [line.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.imshow("Test progress", view)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

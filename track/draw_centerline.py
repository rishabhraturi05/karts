import cv2
import numpy as np
from pathlib import Path

P = Path(__file__).resolve().parent
image = cv2.imread(str(P / "track_reference.png"))
mask = cv2.imread(str(P / "track_mask.png"), 0)
if image is None or mask is None:
    raise FileNotFoundError("Track image/mask")
pts = []


def mouse(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        if mask[y, x] == 255:
            pts.append((x, y))
            print("Added", x, y)
        else:
            print("Click on white track area")


cv2.namedWindow("Centerline")
cv2.setMouseCallback("Centerline", mouse)

while True:
    view = image.copy()
    if pts:
        for pt in pts:
            cv2.circle(view, pt, 3, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts, np.int32)], False, (0, 0, 255), 2)
        if len(pts) > 2:
            cv2.line(view, pts[-1], pts[0], (0, 0, 255), 2)
    cv2.imshow("Centerline", view)
    key = cv2.waitKey(20) & 0xFF
    if key == ord("u") and pts:
        pts.pop()
    elif key == ord("s") and len(pts) >= 3:
        np.save(P / "centerline_points.npy", np.array(pts + [pts[0]], np.float32))
        print("Centerline saved")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

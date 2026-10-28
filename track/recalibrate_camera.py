import cv2
import numpy as np
from pathlib import Path

P = Path(__file__).resolve().parent
CAMERA_INDEX = 0
WARP_W, WARP_H = 1800, 1000
PAD = 40
ref = cv2.imread(str(P / "track_reference.png"))
if ref is None:
    raise FileNotFoundError("track_reference.png")


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera could not open")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5500)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    return cap


params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
det = cv2.aruco.ArucoDetector(d, params)
cap = open_camera(CAMERA_INDEX)
H = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    view = frame.copy()
    corners, ids, _ = det.detectMarkers(frame)
    H = None
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(view, corners, ids)
        found = {int(i): c[0].mean(axis=0) for c, i in zip(corners, ids.flatten())}
        if all(i in found for i in range(4)):
            src = np.float32([found[i] for i in [0, 1, 2, 3]])
            dst = np.float32([
                [PAD, PAD],
                [WARP_W - 1 - PAD, PAD],
                [WARP_W - 1 - PAD, WARP_H - 1 - PAD],
                [PAD, WARP_H - 1 - PAD]
            ])
            H = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(frame, H, (WARP_W, WARP_H))
            cv2.imshow("Warp preview 1260x700", cv2.resize(warped, (1260, 700)))
    cv2.imshow("Corner tags 960x540", cv2.resize(view, (960, 540)))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and H is not None:
        np.save(P / "perspective_matrix.npy", H)
        print("Homography updated only")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

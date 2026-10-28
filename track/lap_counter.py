import cv2
import numpy as np
from pathlib import Path

P = Path(__file__).resolve().parent
CAMERA_INDEX = 0
CAM_W, CAM_H = 1920, 1080
PARALLAX_FACTOR = 1.0 - (0.394 / 12.0)
MIN_CAR_AREA_PX = 800
MAX_CAR_AREA_PX = 9000

CAR_CONFIGS = {
    "Car 1 Red": {
        "ranges": [
            (np.array([0, 120, 50]), np.array([10, 255, 255])),
            (np.array([168, 120, 50]), np.array([179, 255, 255]))
        ],
        "color": (0, 0, 255)
    },
    "Car 2 Blue": {
        "ranges": [
            (np.array([100, 120, 50]), np.array([128, 255, 255]))
        ],
        "color": (255, 0, 0)
    },
    # Optional Car 3 Green: add to CAR_CONFIGS after testing
    # "Car 3 Green": {
    #     "ranges": [(np.array([42, 100, 50]), np.array([82, 255, 255]))],
    #     "color": (0, 255, 0)
    # },
}


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera unavailable")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5500)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    return cap


def detect_blobs(frame, hsv, kernel):
    found = []
    for name, cfg in CAR_CONFIGS.items():
        combined = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in cfg["ranges"]:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if MIN_CAR_AREA_PX <= cv2.contourArea(c) <= MAX_CAR_AREA_PX]
        if valid:
            c = max(valid, key=cv2.contourArea)
            m = cv2.moments(c)
            if m["m00"]:
                u = m["m10"] / m["m00"]
                v = m["m01"] / m["m00"]
                found.append((name, cfg, (u, v), c))
    return found


ref = cv2.imread(str(P / "track_reference.png"))
mask = cv2.imread(str(P / "track_mask.png"), 0)
H = np.load(P / "perspective_matrix.npy")
line = np.load(P / "dense_centerline.npy")
cum = np.load(P / "cumulative_distance.npy")
length = float(np.load(P / "lap_length.npy"))

if ref is None or mask is None or length <= 0:
    raise ValueError("Missing track assets")

h, w = mask.shape
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cap = open_camera(CAMERA_INDEX)

CHECKPOINTS = [25.0, 50.0, 75.0]
MAX_JUMP = 18.0
FINISH_HIGH = 88.0
FINISH_LOW = 12.0

cars = {
    name: {
        "laps": 0,
        "progress": None,
        "previous": None,
        "cp": 0,
        "armed": False,
        "status": "WAITING"
    }
    for name in CAR_CONFIGS
}


def update(name, p):
    c = cars[name]
    prev = c["previous"]
    if prev is None:
        c["previous"] = c["progress"] = p
        c["armed"] = p <= FINISH_LOW
        c["status"] = "RACING" if c["armed"] else "MOVE TO START"
        return
    delta = (p - prev) % 100.0
    if delta > MAX_JUMP:
        c["previous"] = p  # recover next frame; avoid permanent jump lockup
        c["status"] = "JUMP/REVERSE REJECTED"
        return
    c["previous"] = c["progress"] = p
    if not c["armed"]:
        if p <= FINISH_LOW:
            c["armed"] = True
            c["cp"] = 0
            c["status"] = "RACING"
        else:
            c["status"] = "MOVE TO START"
        return
    if p >= prev and c["cp"] < 3 and prev < CHECKPOINTS[c["cp"]] <= p:
        c["cp"] += 1
    if prev >= FINISH_HIGH and p <= FINISH_LOW and p < prev:
        if c["cp"] == 3:
            c["laps"] += 1
            print(name, "LAP", c["laps"])
        else:
            print(name, "finish rejected: checkpoints incomplete")
        c["cp"] = 0
        c["status"] = "RACING"


while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame failed")
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    track = ref.copy()
    cv2.polylines(track, [line.astype(np.int32)], True, (255, 255, 0), 2)
    seen = set()
    for name, cfg, (u, v), contour in detect_blobs(frame, hsv, kernel):
        ug = CAM_W / 2 + (u - CAM_W / 2) * PARALLAX_FACTOR
        vg = CAM_H / 2 + (v - CAM_H / 2) * PARALLAX_FACTOR
        x, y = cv2.perspectiveTransform(np.array([[[ug, vg]]], np.float32), H)[0, 0]
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < w and 0 <= py < h):
            cars[name]["status"] = "OUTSIDE IMAGE"
            continue
        seen.add(name)
        if mask[py, px] != 255:
            cars[name]["status"] = "OFF TRACK"
            continue
        idx = int(np.argmin(np.sum((line - [x, y]) ** 2, axis=1)))
        progress = float(cum[idx] / length * 100)
        update(name, progress)
        cv2.circle(track, (px, py), 8, cfg["color"], -1)
    for j, (name, c) in enumerate(cars.items()):
        status = c["status"] if name in seen else "NOT DETECTED"
        prog = "--" if c["progress"] is None else f"{c['progress']:.1f}%"
        cv2.putText(
            track,
            f"{name} Laps:{c['laps']} Progress:{prog} CP:{c['cp']}/3 {status}",
            (12, 28 + j * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )
    cv2.imshow("HSV camera 960x540", cv2.resize(frame, (960, 540)))
    cv2.imshow("Lap counter 1260x700", cv2.resize(track, (1260, 700)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

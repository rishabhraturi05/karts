import cv2
import numpy as np

CAMERA_INDEX = 0


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


cap = open_camera(CAMERA_INDEX)
last = None
print("Click the centre of a matte car roof; Q exits")


def click(event, x, y, flags, param):
    global last
    if event == cv2.EVENT_LBUTTONDOWN and last is not None:
        hsv = cv2.cvtColor(last, cv2.COLOR_BGR2HSV)
        sx = min(last.shape[1] - 1, int(x * last.shape[1] / 960))
        sy = min(last.shape[0] - 1, int(y * last.shape[0] / 540))
        h, s, v = [int(z) for z in hsv[sy, sx]]
        # Conservative suggested band; red wraps Hue 0/179
        if h <= 10 or h >= 168:
            ranges = [(0, max(10, h + 5)), (max(168, h - 5), 179)]
        else:
            ranges = [(max(0, h - 8), min(179, h + 8))]
        lo_s, hi_s = max(0, s - 50), min(255, s + 50)
        lo_v, hi_v = max(0, v - 60), min(255, v + 40)
        print(f"Clicked HSV: [{h}, {s}, {v}]")
        for lo_h, hi_h in ranges:
            print("lower = np.array([%d, %d, %d])" % (lo_h, lo_s, lo_v))
            print("upper = np.array([%d, %d, %d])" % (hi_h, hi_s, hi_v))
        print("Verify sampled ranges against full frame, shadows and track background.")


cv2.namedWindow("HSV picker 960x540")
cv2.setMouseCallback("HSV picker 960x540", click)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame failed")
        break
    last = frame.copy()
    cv2.imshow("HSV picker 960x540", cv2.resize(frame, (960, 540)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

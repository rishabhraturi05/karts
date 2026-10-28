import cv2

CAMERA_INDEX = 0  # set from camera_detect.py


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
print("Requested 1920x1080@30 MJPG; checking actual frames")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed")
        break
    print(f"Actual frame: {frame.shape[1]}x{frame.shape[0]}", end="\r")
    cv2.imshow("FHDWC300 1080p camera", cv2.resize(frame, (960, 540)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

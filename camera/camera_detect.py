import cv2

for i in range(8):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            print(f"Index {i}: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()

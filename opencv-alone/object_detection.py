import datetime
import time
import cv2
from ultralytics import YOLO
import os

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('No cameras found')
    exit()

frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(capture.get(cv2.CAP_PROP_FPS))
if frame_fps <= 0 or frame_fps > 30:
    frame_fps = 30.0

photos = 'photos'
os.makedirs(photos, exist_ok=True)

recordings = 'recordings'
os.makedirs(recordings, exist_ok=True)

model = YOLO('../yolo_pt/yolo11n.pt')
labels = ['airplane', 'car', 'cat', 'dog', 'person']

is_recording = False
out = None

start_fps = time.time()

while True:
    status, frame = capture.read()
    if not status:
        print('No frames')
        break

    frame = cv2.flip(frame, 1)

    count = 0

    predict = model(frame, stream=True, verbose=False, conf=0.3)
    for f in predict:
        for obj in f.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            cls = int(obj.cls[0])
            label = model.names[cls]

            if label not in labels:
                continue

            if label == 'person': count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, f'{label}', (x1, y1 - 15), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 255, 255), 1)

    cv2.putText(frame, f'persons: {count}', (450, 80), cv2.FONT_HERSHEY_COMPLEX,
                0.8, (255, 255, 255), 1)

    end_fps = time.time()
    fps = round(1.0 / (end_fps - start_fps))
    start_fps = end_fps

    cv2.putText(frame, f'fps: {fps}', (10, 80), cv2.FONT_HERSHEY_COMPLEX,
                0.8, (255, 255, 255), 1)
    cv2.putText(frame, 'II19', (300, 40), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 255, 255), 1)
    current_date_str = datetime.datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    cv2.putText(frame, f'date: {current_date_str}', (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('s'):
        photo_date = datetime.datetime.now().strftime('%d.%m.%Y_%H-%M-%S')
        file_name = f'{photos}/photo_{photo_date}.jpg'
        cv2.imwrite(file_name, frame)
        print(f'Photo saved: {file_name}')

    if is_recording and out is not None:
        out.write(frame)
    if key == ord('r'):
        if not is_recording:
            video_date = datetime.datetime.now().strftime('%d.%m.%Y_%H-%M-%S')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'{recordings}/video_{video_date}.mp4', fourcc, frame_fps,
                                  (frame_width, frame_height))
            is_recording = True
        else:
            is_recording = False

if out is not None:
    out.release()
capture.release()
cv2.destroyAllWindows()
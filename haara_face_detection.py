import cv2
import datetime
import os

savings = 'media'
os.makedirs(savings, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print('No camera')
    exit()

counter = 0

while True:
    status, frame = capture.read()

    if not status:
        print('No frames')
        break

    gray_frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_data = face_cascade.detectMultiScale(gray_frames, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in face_data:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    cv2.imshow(winname='Video', mat=frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        saving_date = datetime.datetime.now().strftime('%d.%m.%Y-%H-%M-%S')
        savings_name = f'{savings}/photo_{saving_date}.jpg'
        cv2.imwrite(savings_name, frame)
        counter += 1
        print(counter)
        cv2.putText(frame, '*saved*', (10, 450), cv2.FONT_HERSHEY_TRIPLEX,
                    1, (255, 255 , 255), 2)
        cv2.imshow('Photo', frame)
        cv2.waitKey(300)

    elif key == ord('`'):
        break

capture.release()
cv2.destroyAllWindows()
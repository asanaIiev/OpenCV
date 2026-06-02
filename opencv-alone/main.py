import cv2
from datetime import datetime

capture = cv2.VideoCapture(0)
if not capture.isOpened(): print('No Cameras Found')

while True:
    status, frames = capture.read()
    if not status: print('No Frames')

    cv2.putText(frames, 'II19', (300, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)

    cv2.putText(frames, 'hello_world("print")', (150, 250),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)

    date_format = datetime.now().strftime('%d.%m.%Y')
    time_format = datetime.now().strftime('%H:%M:%S')
    cv2.putText(frames, f'DATE: {date_format}', (450, 100), cv2.FONT_HERSHEY_TRIPLEX,
                0.6, (255, 255, 255), 1)
    cv2.putText(frames, f'TIME: {time_format}', (450, 125), cv2.FONT_HERSHEY_TRIPLEX,
                0.6, (255, 255, 255), 1)

    cv2.putText(frames, 'asanaliev',
                (450, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)

    cv2.imshow(winname='Video', mat=frames)

    if cv2.waitKey(1) & 0xFF == ord('`'):
        break

capture.release()
cv2.destroyAllWindows()
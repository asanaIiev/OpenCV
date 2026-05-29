import cv2

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print('No camera')
    exit()

status1, frame1 = capture.read()
status2, frame2 = capture.read()

sensitivity = 20

while True:
    diff = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, sensitivity, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    display_frame = frame1.copy()

    motion_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 200:
            continue

        motion_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(display_frame, f"Area {int(area)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, (0, 255, 0), 1)

    cv2.putText(display_frame, f'Sensitivity: {sensitivity}', (10, 60), cv2.FONT_ITALIC,
                0.8, (255, 255, 255), 2)

    cv2.imshow('Motion detector', display_frame)
    cv2.imshow('Threshold', thresh)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): break
    if key == ord('s'):
        image_name = f'media/motion_{cv2.getTickCount()}.jpg'
        image_name2 = f'media/motion_thresh_{cv2.getTickCount()}.jpg'
        cv2.imwrite(image_name2, thresh)
        cv2.imwrite(image_name, display_frame)
        print(image_name)
        print(image_name2)

    if key == ord('+') or key == ord('='):
        sensitivity = min(sensitivity + 5, 100)
        print(f'Sensitivity: {sensitivity}')
    if key == ord('-'):
        sensitivity = max(sensitivity - 5, 5)
        print(f'Sensitivity: {sensitivity}')

    frame1 = frame2
    status1, frame2 = capture.read()

    if not status1:
        print('No frames')
        break

capture.release()
cv2.destroyAllWindows()
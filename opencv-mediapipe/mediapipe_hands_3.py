import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('No camera')
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=3,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)
draw_hands = mp.solutions.drawing_utils

current_mode = None
prev_gray = None

while True:
    status, frame = capture.read()
    if not status:
        print('No frames')
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_hands = hands.process(rgb_frame)

    if current_mode == 'blue':
        display_frame = cv2.GaussianBlur(frame, (51, 51), 0)

    elif current_mode == 'black':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
            display_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        else:
            display_frame = frame.copy()
        prev_gray = gray

    elif current_mode == 'gray':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif current_mode == 'white':
        display_frame = frame.copy()

    else:
        display_frame = frame.copy()

    rectangles = [
        ('blue', (255,0,0), 10),
        ('black', (0,0,0), 180),
        ('gray', (128,128,128), 350),
        ('white', (255,255,255), 520)
    ]
    for mode, color, color_x in rectangles:
        cv2.rectangle(display_frame, (color_x, 10), (color_x+100, 100), color, -1)
        if current_mode == mode:
            cv2.rectangle(display_frame, (color_x-3, 7), (color_x+103, 103), (0, 255, 0), 1)

    if processed_hands.multi_hand_landmarks:
        for hand in processed_hands.multi_hand_landmarks:
            draw_hands.draw_landmarks(display_frame, hand, mp_hands.HAND_CONNECTIONS)

            x8, y8 = round(hand.landmark[8].x * w), round(hand.landmark[8].y * h)

            if 10 < y8 < 100:
                for mode, color, color_x in rectangles:
                    if color_x < x8 < color_x + 100:
                        current_mode = mode
                        break

    cv2.imshow('Camera', display_frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
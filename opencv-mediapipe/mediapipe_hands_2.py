import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('No camera')
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
draw_hands = mp.solutions.drawing_utils

COLOR_LIGHTBLUE_RGB = (0, 255, 255)
while True:
    status, frame = capture.read()
    if not status:
        print('No frames')
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_hands = hands.process(rgb_frame)

    if processed_hands.multi_hand_landmarks:
        for hand in processed_hands.multi_hand_landmarks:
            draw_hands.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            index_up = hand.landmark[8].y < hand.landmark[6].y
            middle_up = hand.landmark[12].y < hand.landmark[10].y
            ring_up = hand.landmark[16].y < hand.landmark[14].y
            pinky_up = hand.landmark[20].y < hand.landmark[18].y

            is_horizontal = abs(hand.landmark[5].x - hand.landmark[17].x) < 0.05
            index_horizontal = abs(hand.landmark[8].x - hand.landmark[5].x) > 0.1
            middle_horizontal = abs(hand.landmark[12].x - hand.landmark[9].x) > 0.1

            up_fingers = 0
            curved_side_fingers = 0

            for dot_id, xyz in enumerate(hand.landmark):
                if dot_id in [8, 12, 16, 20] and hand.landmark[dot_id].y < hand.landmark[dot_id - 1].y:
                    up_fingers += 1

                if dot_id in [8, 12, 16, 20] and abs(hand.landmark[dot_id].x - hand.landmark[dot_id - 2].x) > 0.05:
                    curved_side_fingers += 1

            detected_letter = None

            if curved_side_fingers == 4 and is_horizontal:
                detected_letter = "C"

            elif up_fingers == 4 and not is_horizontal:
                detected_letter = "B"

            elif index_up and not middle_up and not ring_up and not pinky_up:
                detected_letter = "D"

            elif not index_up and middle_up and ring_up and pinky_up:
                detected_letter = "F"

            elif index_horizontal and not middle_horizontal:
                detected_letter = "G"

            elif index_horizontal and middle_horizontal:
                detected_letter = "H"

            elif up_fingers == 0 and hand.landmark[8].y > hand.landmark[6].y and hand.landmark[8].y < hand.landmark[5].y:
                detected_letter = "E"

            elif up_fingers == 0:
                detected_letter = "A"

            if detected_letter:
                cv2.putText(frame, f'Letter: {detected_letter}', (250, 40), cv2.FONT_HERSHEY_COMPLEX,
                            1, COLOR_LIGHTBLUE_RGB[::-1], 2)

    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'): break

capture.release()
cv2.destroyAllWindows()
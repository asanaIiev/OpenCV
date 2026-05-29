import mediapipe as mp
import cv2

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('No camera')
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
draw_hands = mp.solutions.drawing_utils

canvas_points = []

COLOR_LIGHTBLUE_RGB = (0, 255, 255)
while True:
    status, frame = capture.read()
    if not status: break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_hands = hands.process(rgb_frame)

    if processed_hands.multi_hand_landmarks:
        for hand_id, hand in enumerate(processed_hands.multi_hand_landmarks):
            draw_hands.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            x, y = round(hand.landmark[0].x * w), round(hand.landmark[0].y * h)
            cv2.putText(frame, f'{hand_id+1}', (x, y+30), cv2.FONT_HERSHEY_TRIPLEX,
                        0.8, COLOR_LIGHTBLUE_RGB[::-1], 1)

            x8, y8 = round(hand.landmark[8].x * w), round(hand.landmark[8].y * h)
            if cv2.waitKey(1) & 0xFF == ord('p'):
                canvas_points.append((x8, y8))
            else:
                if len(canvas_points) > 0 and canvas_points[-1] is not None:
                    canvas_points.append(None)

            for i in range(1, len(canvas_points)):
                if canvas_points[i-1] is not None and canvas_points[i] is not None:
                    cv2.line(frame, canvas_points[i-1], canvas_points[i], COLOR_LIGHTBLUE_RGB[::-1], 5)

            total_fingers = 0
            for dot_id, xyz in enumerate(hand.landmark):
                x, y = round(xyz.x * w), round(xyz.y * h)
                cv2.putText(frame, f'{dot_id}', (x, y), cv2.FONT_HERSHEY_TRIPLEX,
                            0.6, (0,0,0), 2)

                fingers = []
                if dot_id in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (x, y), 7, COLOR_LIGHTBLUE_RGB[::-1], -1)

                if dot_id in [8, 12, 16, 20] and hand.landmark[dot_id].y < hand.landmark[dot_id-1].y:
                    fingers.append(1)

                elif dot_id == 4:
                    if abs(hand.landmark[4].x - hand.landmark[2].x) > 0.05:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else: fingers.append(0)

                total_fingers += sum(fingers)
            cv2.putText(frame, f'Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (255, 255, 255), 1)

        cv2.putText(frame, f'Detected hands: {len(processed_hands.multi_hand_landmarks)}',
                    (10, 40), cv2.FONT_HERSHEY_COMPLEX,
                0.8, (255, 255, 255), 1)

    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): canvas_points.clear()

capture.release()
cv2.destroyAllWindows()
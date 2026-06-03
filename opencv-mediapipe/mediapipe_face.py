import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('No camera')
    exit()

mp_faces = mp.solutions.face_mesh
faces = mp_faces.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)
draw_faces = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)
draw_hands = mp.solutions.drawing_utils

blink_count = 0
eye_closed = False

while True:
    status, frame = capture.read()
    if not status:
        print('No frames')
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_faces = faces.process(rgb_frame)

    processed_hands = hands.process(rgb_frame)

    if processed_faces.multi_face_landmarks:
        for face in processed_faces.multi_face_landmarks:
            draw_faces.draw_landmarks(frame, face, mp_faces.FACEMESH_TESSELATION)

            cv2.putText(frame, f'Faces: {len(processed_faces.multi_face_landmarks)}', (10, 40),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 1)

            y_upper_lip, y_lower_lip = face.landmark[13].y, face.landmark[14].y

            if y_lower_lip - y_upper_lip > 0.02:
                label = 'Mouth status: OPEN'
            else: label = 'Mouth status: CLOSE'
            cv2.putText(frame, f'{label}', (10, 70),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 1)

            left_eye_h = face.landmark[145].y - face.landmark[159].y
            right_eye_h = face.landmark[374].y - face.landmark[386].y

            if left_eye_h < 0.017 and right_eye_h < 0.017:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else: eye_closed = False
            cv2.putText(frame, f'Blinks: {blink_count}', (10, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 1)

            # cv2.putText(frame, f'L: {left_eye_h:.4f} R: {right_eye_h:.4f}', (10, 130),
            #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)

    if processed_hands.multi_hand_landmarks:
        for hand in processed_hands.multi_hand_landmarks:
            draw_hands.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'): break

capture.release()
cv2.destroyAllWindows()
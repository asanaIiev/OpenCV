import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 1. Настройка стилей отрисовки (Цвета в формате BGR)
# Стиль для точек (например, красные точки)
landmark_style = mp_drawing.DrawingSpec(
    color=(0, 255, 0),      # Красный цвет (B=0, G=0, R=255)
    thickness=2,            # Толщина контура точки
    circle_radius=4         # Радиус точки
)

# Стиль для связывающих линий (например, жёлтые линии)
connection_style = mp_drawing.DrawingSpec(
    color=(0, 255, 0),    # Жёлтый цвет (B=0, G=255, R=255)
    thickness=3             # Толщина линии
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # 2. Передаем наши стили в draw_landmarks
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,      # Цвет и размер точек
            connection_drawing_spec=connection_style   # Цвет и толщина линий
        )

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
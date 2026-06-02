import time
from ultralytics import YOLO
import cv2
import streamlit as st
import numpy as np
import tempfile

st.sidebar.title('OpenCV')
models = st.sidebar.selectbox('Choose YOLO Model',
                              ['yolo_pt/yolov5nu.pt', 'yolo_pt/yolov8n.pt',
                               'yolo_pt/yolo11n.pt', 'yolo_pt/yolo26n.pt'])
mode_type = st.sidebar.radio('Mode type', ['Photo', 'Video', 'Camera'])

model = YOLO(models)
labels = list(model.names.values())

select_labels = st.sidebar.multiselect('Choose classes for detection', labels,
                       default=['person'])
sidebar_button = st.sidebar.button('Detect')

def predict(frame):
    result = model(frame, stream=True, verbose=False, conf=0.5)
    counts = {}

    for f in result:
        for obj in f.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            cls = int(obj.cls[0])
            label = model.names[cls]
            counts[label] = counts.get(label, 0) + 1

            if not label in select_labels: continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f'{label}', (x1, y1-15), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 255, 255), 2)
    return frame, counts

if mode_type == 'Photo':
    uploaded_image = st.file_uploader('Upload an image', type=['png', 'jpeg', 'jpg'])
    if uploaded_image and sidebar_button:
        byte_image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(byte_image, cv2.IMREAD_COLOR)
        detection, quantity = predict(image)
        rgb_image = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
        st.image(rgb_image, 'Detected image', use_container_width=True)
        st.write(f'Detected objects: {quantity}')

elif mode_type == 'Video':
    uploaded_video = st.file_uploader('Upload a video', type=['mp4'])
    if uploaded_video and sidebar_button:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        video_placeholder = st.empty()
        text_placeholder = st.empty()

        capture = cv2.VideoCapture(temp_file.name)
        while capture.isOpened():
            status, frames = capture.read()
            if not status: break

            detection, quantity = predict(frames)
            rgb_video = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_video)

            text_placeholder.json(quantity)

        capture.release()

elif mode_type == 'Camera' and sidebar_button:
    start_fps = 0

    video_placeholder = st.empty()
    text_placeholder = st.empty()

    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        status, frames = capture.read()
        if not status:
            break

        end_fps = time.time()
        fps = round(1.0 / (end_fps - start_fps))
        start_fps = end_fps

        detection, quantity = predict(frames)
        rgb_frame = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame)

        detected_objects = [key for key in quantity.keys() if key in select_labels]
        total_objects = sum(value for key, value in quantity.items() if key in select_labels)
        text_placeholder.markdown(
            f"**FPS**: {fps} | "
            f"**Objects quantity**: {total_objects} | "
            f"**Classes**: {', '.join(detected_objects)}"
        )

    capture.release()

else:
    st.info('Choose mode, and classes to detect')
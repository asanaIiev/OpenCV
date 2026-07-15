# import cv2
# from inference_sdk import InferenceHTTPClient
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
# client = InferenceHTTPClient.init(
#     api_url='https://serverless.roboflow.com',
#     api_key=os.getenv('API_KEY')
# )
#
# capture = cv2.VideoCapture(0)
# if not capture.isOpened():
#     print('No camera')
#     exit()
#
# while True:
#     status, frame = capture.read()
#     if not status:
#         print('No frames left')
#         break
#
#     result = client.infer(frame, model_id=os.getenv('MODEL_ID'))
#     for obj in result.get('predictions', []):
#         x, y, w, h = obj['x'], obj['y'], obj['width'], obj['height']
#         confidence = obj['confidence']
#         cls = obj['class']
#
#         x1, y1, x2, y2 = map(int, [x - w / 2, y - h / 2, x + w / 2, y + h / 2])
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
#
#     frame = cv2.resize(frame, (640, 480))
#     cv2.imshow('Camera', frame)
#     if cv2.waitKey(1) & 0xff == ord('q'): break
#
# capture.release()
# cv2.destroyAllWindows()


import cv2
import os
from dotenv import load_dotenv
from inference import get_model

load_dotenv()
model = get_model(model_id=os.getenv('MODEL_ID'), api_key=os.getenv('API_KEY'))

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('No camera')
    exit()

while True:
    status, frame = capture.read()
    if not status:
        print('No frames left')
        break

    sheep_count = 0
    cow_count = 0

    result = model.infer(frame)[0]

    for obj in result.predictions:
        x1, y1, x2, y2 = map(
            int, [obj.x - obj.width / 2, obj.y - obj.height / 2, obj.x + obj.width / 2, obj.y + obj.height / 2]
        )
        cls = obj.class_name
        conf = obj.confidence

        if cls.lower() == 'sheep':
            sheep_count += 1
        elif cls.lower() == 'cow':
            cow_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.putText(frame, f'{cls}: {conf * 100:.2f}%', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2 )

    cv2.putText(frame, f'SHEEP COUNT: {sheep_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 1)

    cv2.putText(frame, f'COW COUNT: {cow_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 1)


    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
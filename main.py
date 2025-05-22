import cv2
from ultralytics import YOLO
import numpy as np

# Cargar Modelo de Detección de Pose
pose_detector = YOLO("model/yolo11n-pose.pt")

# Cargar Video
cap = cv2.VideoCapture("videos/example2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = pose_detector(frame)[0]
    boxes = result.boxes.data.tolist()
    keypoints = result.keypoints.data.tolist()

    for bbox in boxes:
        x1, y1, x2, y2, score, class_id = bbox
        if score > 0.8:
            cv2.putText(frame, f'Score: {score:.2f}', (int(x1) + 10, int(y2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    for person_keypoint in keypoints:
        for i, keypoint in enumerate(person_keypoint):
            x, y, conf = keypoint
            if conf > 0.7:
                cv2.putText(frame, str(i), (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        if all(k[2] > 0.7 for k in [person_keypoint[11], person_keypoint[13], person_keypoint[15]]):
            # Pierna Izquierda
            left_leg = np.array((person_keypoint[11], person_keypoint[13], person_keypoint[15]))
            pts_left = left_leg[:, :2].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_left], False, (255, 226, 79), 2)

        if all(k[2] > 0.7 for k in [person_keypoint[11], person_keypoint[13], person_keypoint[15]]):
            # Pierna Derecha
            right_leg = np.array((person_keypoint[12], person_keypoint[14], person_keypoint[16]))
            pts_right = right_leg[:, :2].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_right], False, (255, 226, 79), 2)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Deteccion de Pose", frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
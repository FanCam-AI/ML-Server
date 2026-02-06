import cv2
import matplotlib.pyplot as plt
import supervision as sv
from super_gradients.training import models
import logging
import numpy as np
import torch

logging.getLogger("super_gradients").setLevel(logging.WARNING)



class FaceDetection:
    def __init__(self, detection_model_path, detection_model_name):
        model_arch = 'yolo_nas_m'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if detection_model_name == "person":
            self.model = models.get(
                model_arch,
                num_classes=1,
                checkpoint_path=detection_model_path
                ).to(device)

        elif detection_model_name == "animal":
            self.model = models.get(
                model_arch,
                num_classes=2,
                checkpoint_path=detection_model_path
            ).to(device)
        self.detection_model_name = detection_model_name

    def detect_face(self, image, visualize=False):
        if self.detection_model_name == "person":
            confidence_threshold = 0.5
            try:
                result = self.model.predict(image, conf=confidence_threshold)  # 수정됨
            except ValueError as e:
                # super_gradients에서 얼굴이 없을 때 발생하는 예외
                if "Input None not supported" in str(e):
                    return np.array([])  # 빈 bbox 배열 반환
                else:
                    raise  # 다른 오류는 그대로 올림
            detections = sv.Detections(
                xyxy=result.prediction.bboxes_xyxy,
                confidence=result.prediction.confidence,
                class_id=result.prediction.labels.astype(int)
            )

            if visualize:
                box_annotator = sv.BoxAnnotator()
                label_annotator = sv.LabelAnnotator()

                annotated_image = image.copy()
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

                sv.plot_image(annotated_image)


            return detections.xyxy


        elif self.detection_model_name == "animal":
            confidence_threshold = 0.5
            try:
                result = self.model.predict(image, conf=confidence_threshold)  # 수정됨
            except ValueError as e:
                # super_gradients에서 얼굴이 없을 때 발생하는 예외
                if "Input None not supported" in str(e):
                    return np.array([])  # 빈 bbox 배열 반환
                else:
                    raise  # 다른 오류는 그대로 올림
            detections = sv.Detections(
                xyxy=result.prediction.bboxes_xyxy,
                confidence=result.prediction.confidence,
                class_id=result.prediction.labels.astype(int)
            )

            if visualize:
                box_annotator = sv.BoxAnnotator()
                label_annotator = sv.LabelAnnotator()

                annotated_image = image.copy()
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

                sv.plot_image(annotated_image)

            return detections.xyxy


        else:
            pass

    @staticmethod
    def extract_face(face_boxes, image, visualize=False):
        resized_face_arr = []
        face_box_arr = []

        if len(face_boxes) == 0:

            return None, None

        else:
            img_h, img_w = image.shape[:2]
            for face_box in face_boxes:
                x1, y1, x2, y2 = map(int, face_box)  # 좌표 정수 변환
                w, h = x2 - x1, y2 - y1  # 너비, 높이 계산

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                w, h = x2 - x1, y2 - y1

                face = image[y1:y1 + h, x1:x1 + w]

                resized_face = cv2.resize(face, (320, 320))


                face_box_arr.append((x1, y1, w, h))
                resized_face_arr.append(resized_face)


            if visualize:
                for resized_face in resized_face_arr:
                    plt.imshow(resized_face)
                    plt.axis("off")  # 축 숨기기 (선택 사항)
                    plt.show()


            return resized_face_arr, face_box_arr


    @staticmethod
    def find_closest_bbox_to_center(image, bboxes):
        img_h, img_w = image.shape[:2]
        img_center = np.array([img_w / 2, img_h / 2])

        min_dist = float('inf')
        closest_bbox = None

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            dist = np.linalg.norm(bbox_center - img_center)

            if dist < min_dist:
                min_dist = dist
                closest_bbox = bbox

        return [closest_bbox] if closest_bbox is not None else list()

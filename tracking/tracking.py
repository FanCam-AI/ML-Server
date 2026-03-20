import cv2
import pyheif
from PIL import Image
import numpy as np
import os
from processing import AutoRotatedVideoProcessor

class Tracking:
    def __init__(self, tracker, video_path, query_image_paths, face_detection, face_recognition, detection_model_name, redis_client):
        self.fit_to = 'height'
        self.tracker = tracker
        self.video_path = video_path
        self.fourcc =cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.face_detector = face_detection
        self.face_recognizer = face_recognition
        self.top_bottom_list = []
        self.left_right_list = []
        self.progress = int()
        self.ema_center = None
        self.ema_size = None
        self.query_images = [self.load_image(p) for p in query_image_paths]
        self.query_embeddings = []
        self.processor = AutoRotatedVideoProcessor(self.video_path)
        self.detection_model_name = detection_model_name
        self.isPortrait = None
        self.redis_client = redis_client


        max_height = 2160  # 4K의 높이
        max_width = 3840  # 4K의 너비
        aspect_ratio = 9 / 16

        cap = cv2.VideoCapture(video_path)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_height = min(frame_height, max_height)
        target_width = int(target_height * aspect_ratio)

        if target_width > max_width:

            target_width = max_width
            target_height = int(target_width / aspect_ratio)

        processor = AutoRotatedVideoProcessor(self.video_path)
        if processor.is_portrait():
            print("세로 영상")
            self.output_size = (target_width, target_height-100)
            self.isPortrait = True
        else:
            print("가로 영상")
            self.output_size = (target_width, target_height-100)
            self.isPortrait = False

        print(self.output_size)
        self.output_path = f"{os.path.splitext(video_path)[0]}_output.mp4"


    def normal_mode(self, start_time, end_time, user_id, drag_box, visualize=False):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("VideoCapture가 정상적으로 열리지 않았습니다.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        if start_time >= duration:
            print(f"[WARN] start_time {start_time}s is longer than video duration {duration:.2f}s")
            start_time = max(0, duration - 1)
        if end_time > duration:
            end_time = duration
        start_frame = round(fps * start_time)
        end_frame = round(fps * end_time)
        current_frame = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        out = cv2.VideoWriter(self.output_path, self.fourcc, cap.get(cv2.CAP_PROP_FPS), self.output_size)
        ret, img = cap.read()
        img = self.processor.rotate_frame(img)
        h_img, w_img = img.shape[:2]
        drag_resized_box = self.resize_tracker_bbox(drag_box, w_img, h_img)
        out_w = int(drag_resized_box[2])
        out_h = int(drag_resized_box[3])
        self.output_size = (out_w, out_h)

        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = self.processor.rotate_frame(img)
            resized_box = drag_resized_box
            avg_height_range, avg_width_range, left, right, top, bottom = self.compute_average_move(resized_box)
            result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

            result_img = cv2.resize(result_img, self.output_size)
            if visualize:
                pt1 = (int(left), int(top))
                pt2 = (int(right), int(bottom))
                cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

                cv2.imshow('img', img)
                cv2.imshow('result', result_img)
                cv2.waitKey(1)

            out.write(result_img)
            current_frame += 1

            if current_frame == end_frame:
                break
        if visualize:
            cv2.destroyAllWindows()

        cap.release()
        out.release()
        self.redis_client.set(f"job_progress:{user_id}", 90, ex=600)




    def precision_mode(self, start_time, end_time, user_id, drag_box, visualize=False):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("VideoCapture가 정상적으로 열리지 않았습니다.")
        threshold = None
        if self.detection_model_name == "person":
            threshold = 0.3
        elif self.detection_model_name == "animal":
            threshold = 0.3

        print(threshold)
        init_threshold_calculated = False
        resized_box = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("precision_mode fps:", fps)

        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        if start_time >= duration:
            print(f"[WARN] start_time {start_time}s is longer than video duration {duration:.2f}s")
            start_time = max(0, duration - 1)
        if end_time > duration:
            end_time = duration
        start_frame = round(fps * start_time)
        end_frame = round(fps * end_time)
        current_frame = start_frame
        total_frame = end_frame - start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        init_count = start_frame
        resized_init_rect = list()
        baseline_distances = list()
        skip_distance_check = bool()

        for image in self.query_images:
            query_face_detection_boxes = self.face_detector.detect_face(image=image, visualize=False)
            closet_center_query_face_bbox = self.face_detector.find_closest_bbox_to_center(image=image,bboxes=query_face_detection_boxes)
            query_rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            query_faces, _ = self.face_detector.extract_face(closet_center_query_face_bbox, query_rgb_img, visualize=False)

            if query_faces is not None:
                for face in query_faces:
                    embedding = self.face_recognizer.get_embedding(face)
                    self.query_embeddings.append(embedding)

        while True:
            ret, init_img = cap.read()
            min_distance = None
            if not ret:
                break
            print(init_count)
            if len(self.query_embeddings) == 0:
                break
            if init_count == end_frame / 2:
                break
            if init_count % 5 == 0:
                init_img = self.processor.rotate_frame(init_img)
                face_detection_boxes = self.face_detector.detect_face(init_img, visualize=False)
                rgb_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
                extracted_faces_arr, extracted_faces_box_arr = self.face_detector.extract_face(face_detection_boxes, rgb_img, visualize=False)
                processed_frame = init_count - start_frame
                self.progress = int((processed_frame  / total_frame) * 90)
                self.redis_client.set(f"job_progress:{user_id}", self.progress, ex=3600)

                if extracted_faces_arr is not None:
                    for idx, extracted_face in enumerate(extracted_faces_arr):
                        extracted_face_embedding = self.face_recognizer.get_embedding(extracted_face)

                        l2_distances = [
                            self.face_recognizer.compute_similarity_distance(
                                query_face_embedding=query_embedding,
                                registered_face_embedding=extracted_face_embedding
                            )
                            for query_embedding in self.query_embeddings
                        ]
                        min_distance = min(l2_distances)
                        print(l2_distances)
                        if not init_threshold_calculated and 0.01 <= min_distance < 0.15:
                            min_distance_rounded = round(min_distance, 2)
                            if baseline_distances.count(min_distance_rounded) < 2:
                                baseline_distances.append(min_distance_rounded)

                        if not init_threshold_calculated and len(baseline_distances) == 15:
                            sorted_distances = sorted(baseline_distances)

                            mid_index = len(sorted_distances) // 2
                            middle_two_avg = (sorted_distances[mid_index - 1] + sorted_distances[mid_index]) / 2
                            threshold = round(middle_two_avg, 3)

                            print("재조정된 1차 threshold:", threshold)
                            init_threshold_calculated = True



                        if min_distance < threshold:
                            init_rect = extracted_faces_box_arr[idx]
                            init_rect = list(init_rect)
                            h_img, w_img = init_img.shape[:2]
                            resized_init_rect = self.resize_tracker_bbox(init_rect, w_img, h_img)
                            scale = 0.6
                            portrait_scale = 0.4
                            print("Init bbox:", resized_init_rect)
                            print("Image shape:", init_img.shape)

                            if self.isPortrait:
                                portrait_out_w = int(resized_init_rect[2] * portrait_scale)
                                portrait_out_h = int(resized_init_rect[3] * portrait_scale)
                                self.output_size = (portrait_out_w, portrait_out_h)
                            else:
                                out_w = int(resized_init_rect[2] * scale)
                                out_h = int(resized_init_rect[3] * scale)
                                self.output_size = (out_w, out_h)

                            print("🔥 Dynamic Output Size:", self.output_size)
                            break

                else:
                    pass

                if min_distance is not None:
                    if min_distance < threshold:
                        print(min_distance)
                        break

            init_count += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(self.output_path, self.fourcc, cap.get(cv2.CAP_PROP_FPS), self.output_size)
        ret, img = cap.read()
        img = self.processor.rotate_frame(img)
        center_rect = self.get_center_bbox(img, box_width=self.output_size[0], box_height=self.output_size[1])
        h_img, w_img = img.shape[:2]
        center_resized_box = self.resize_tracker_bbox(center_rect, w_img, h_img)
        tracker = self.create_opencv_tracker()
        activate_tracking = False
        first_init_tracker = False
        threshold_calculated = False

        if  init_count - start_frame <= 10 and len(resized_init_rect) > 0:
            activate_tracking = True
            first_init_tracker = True
            tracker.init(img, resized_init_rect)
            print(start_frame)
            print(init_count)
            print("target detected in 10 frame")


        else:
            print(start_frame)
            print(init_count)
            print("target didn't detected in 10 frame")


        if drag_box is not None:
            drag_resized_box = self.resize_tracker_bbox(drag_box, w_img, h_img)
            out_w = int(drag_resized_box[2])
            out_h = int(drag_resized_box[3])
            self.output_size = (out_w, out_h)



        while True:
            print(current_frame)
            ret, img = cap.read()
            if not ret:
                break
            img = self.processor.rotate_frame(img)

            if current_frame > init_count:
                processed_frame = current_frame - start_frame
                self.progress = int((processed_frame / total_frame) * 90)
                self.redis_client.set(f"job_progress:{user_id}", self.progress, ex=3600)

            if current_frame == init_count and first_init_tracker == False:
                if activate_tracking:
                    del tracker
                    tracker = self.create_opencv_tracker()
                    tracker.init(img, resized_init_rect)

                elif activate_tracking == False and len(resized_init_rect) > 0:
                    tracker = self.create_opencv_tracker()
                    tracker.init(img, resized_init_rect)
                    activate_tracking = True

            if not ret:
                break

            if activate_tracking:
                if current_frame % 10 == 0:
                    face_detection_boxes = self.face_detector.detect_face(img, visualize=False)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    extracted_faces_arr, extracted_faces_box_arr = self.face_detector.extract_face(face_detection_boxes,
                                                                                                   rgb_img, visualize=False)
                    if extracted_faces_arr is not None:
                        for idx, extracted_face in enumerate(extracted_faces_arr):
                            extracted_face_embedding = self.face_recognizer.get_embedding(extracted_face)
                            l2_distances = [
                                self.face_recognizer.compute_similarity_distance(
                                    query_face_embedding=query_embedding,
                                    registered_face_embedding=extracted_face_embedding
                                )
                                for query_embedding in self.query_embeddings
                            ]
                            min_distance = min(l2_distances)
                            print(l2_distances)


                            if not threshold_calculated and  0.01 <= min_distance < 0.15:
                                min_distance_rounded = round(min_distance, 2)
                                if baseline_distances.count(min_distance_rounded) < 2:
                                    baseline_distances.append(min_distance_rounded)

                            if not threshold_calculated and len(baseline_distances) == 30:
                                sorted_distances = sorted(baseline_distances)

                                mid_index = len(sorted_distances) // 2
                                middle_two_avg = (sorted_distances[mid_index - 1] + sorted_distances[mid_index]) / 2
                                threshold = round(middle_two_avg, 3)

                                print("재조정된 threshold:", threshold)
                                threshold_calculated = True

                            if min_distance < threshold:
                                print("트렉커 Relocalizing , min_distance:", min_distance)
                                localizing_rect = extracted_faces_box_arr[idx]
                                localizing_rect = list(localizing_rect)
                                h_img, w_img = img.shape[:2]
                                resized_localizing_rect = self.resize_tracker_bbox(localizing_rect, w_img, h_img)


                                del tracker
                                tracker = self.create_opencv_tracker()
                                tracker.init(img, resized_localizing_rect)
                                skip_distance_check = True
                                break


                if self.is_dark_frame(img) and activate_tracking:
                    print('this frame is black')
                    box = resized_box

                else:
                    success, box = self.tracker.update(frame=img, distance_threshold=50, skip_distance_check=skip_distance_check)
                    skip_distance_check = False

                h_img, w_img = img.shape[:2]
                resized_box = self.resize_tracker_bbox(box, w_img, h_img)

            if activate_tracking == False:
                resized_box = center_resized_box

            avg_height_range, avg_width_range, left, right, top, bottom = self.compute_average_move(resized_box)

            result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

            result_img = cv2.resize(result_img, self.output_size)
            if visualize:
                pt1 = (int(left), int(top))
                pt2 = (int(right), int(bottom))
                cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

                cv2.imshow('img', img)
                cv2.imshow('result', result_img)
                cv2.waitKey(1)


            out.write(result_img)
            current_frame += 1

            if current_frame == end_frame:
                break
        if visualize:
            cv2.destroyAllWindows()
            # release everything
        cap.release()
        out.release()
        self.redis_client.set(f"job_progress:{user_id}", 90, ex=600)

    def resize_tracker_bbox(self, bbox, img_width, img_height):
        target_w, target_h = self.output_size
        x, y, w, h = bbox  # 기존 bbox

        # 기존 bbox의 x 중심
        center_x = x + w / 2

        # 좌우는 중심 기준
        new_x = int(center_x - target_w / 2)

        # y는 상단 고정
        new_y = int(y)

        # 이미지 경계 보정 (x)
        if new_x < 0:
            new_x = 0
        elif new_x + target_w > img_width:
            new_x = img_width - target_w

        # 이미지 경계 보정 (y - 아래로만 체크)
        if new_y + target_h > img_height:
            new_y = img_height - target_h

        return new_x, new_y, target_w, target_h

    def compute_average_move(self, box):
        left, top, w, h = [int(v) for v in box]
        right = left + w
        bottom = top + h

        # save sizes of image
        self.top_bottom_list.append(np.array([top, bottom]))
        self.left_right_list.append(np.array([left, right]))

        # use recent 10 elements for crop (window_size=10)
        if len(self.top_bottom_list) > 50:
            del self.top_bottom_list[0]
            del self.left_right_list[0]

        # compute moving average
        avg_height_range = np.mean(self.top_bottom_list, axis=0).astype(int)
        avg_width_range = np.mean(self.left_right_list, axis=0).astype(int)
        avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)])  # (x, y)

        # compute scaled width and height
        avg_height = (avg_height_range[1] - avg_height_range[0])
        avg_width = (avg_width_range[1] - avg_width_range[0])

        # compute new scaled ROI
        avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
        avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])

        # fit to output aspect ratio
        if self.fit_to == 'width':
            avg_height_range = np.array([
                avg_center[1] - avg_width * self.output_size[1] / self.output_size[0] / 2,
                avg_center[1] + avg_width * self.output_size[1] / self.output_size[0] / 2
            ]).astype(int).clip(0, 9999)

            avg_width_range = avg_width_range.astype(int).clip(0, 9999)
        elif self.fit_to == 'height':
            avg_height_range = avg_height_range.astype(int).clip(0, 9999)

            avg_width_range = np.array([
                avg_center[0] - avg_height * self.output_size[0] / self.output_size[1] / 2,
                avg_center[0] + avg_height * self.output_size[0] / self.output_size[1] / 2
            ]).astype(int).clip(0, 9999)



        return avg_height_range, avg_width_range, left, right, top, bottom


    def compute_exponential_average_move(self, box, alpha=0.3):
        left, top, w, h = [int(v) for v in box]
        right = left + w
        bottom = top + h

        # Calculate current center and size… 일: - `timm.cr
        center_x = left + w / 2
        center_y = top + h / 2
        width = w
        height = h

        # Initialize EMA trackers on first call
        if getattr(self, 'ema_center', None) is None:
            self.ema_center = np.array([center_x, center_y], dtype=np.float32)
            self.ema_size = np.array([width, height], dtype=np.float32)
        else:
            self.ema_center = alpha * np.array([center_x, center_y], dtype=np.float32) + \
                              (1 - alpha) * self.ema_center
            self.ema_size = alpha * np.array([width, height], dtype=np.float32) + \
                            (1 - alpha) * self.ema_size

        # Aspect ratio correction
        target_w, target_h = self.output_size
        aspect_ratio = target_w / target_h
        ema_w, ema_h = self.ema_size

        if self.fit_to == 'width':
            ema_h = ema_w / aspect_ratio
        elif self.fit_to == 'height':
            ema_w = ema_h * aspect_ratio

        # Compute final ROI ranges
        cx, cy = self.ema_center
        avg_width_range = np.round([cx - ema_w / 2, cx + ema_w / 2]).astype(int).clip(0, 9999)
        avg_height_range = np.round([cy - ema_h / 2, cy + ema_h / 2]).astype(int).clip(0, 9999)

        return avg_height_range, avg_width_range, left, right, top, bottom

    @staticmethod
    def get_center_bbox(img, box_width, box_height):
        height, width = img.shape[:2]
        x = (width - box_width) // 2
        y = (height - box_height) // 2

        return x, y, box_width, box_height

    @staticmethod
    def is_dark_frame(image, brightness_threshold=30, dark_ratio_threshold=0.98):
        """
        사람이 보기에 어두운(검정에 가까운) 이미지를 판별
        - brightness_threshold: 이 값 이하를 '검정'으로 간주 (0~255)
        - dark_ratio_threshold: 전체 픽셀 중 몇 %가 검정이면 '검정 프레임'으로 판단
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_pixels = np.sum(gray < brightness_threshold)
        total_pixels = gray.size
        dark_ratio = dark_pixels / total_pixels

        return dark_ratio > dark_ratio_threshold


    def create_opencv_tracker(self):
        return self.tracker

    @staticmethod
    def read_heic_img(path):
        heif_file = pyheif.read(path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        return np.array(image)


    def load_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.heic':
            return self.read_heic_img(path)
        else:
            return cv2.imread(path)
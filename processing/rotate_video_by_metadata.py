import cv2
import ffmpeg

class AutoRotatedVideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.rotation = self.get_rotation()
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Can't open video file: {video_path}")

        # 영상 정보 읽기
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 회전 후 가로/세로가 바뀌는 경우 처리
        if self.rotation in [90, 270]:
            self.output_size = (self.height, self.width)
        else:
            self.output_size = (self.width, self.height)


        print(f"[INFO] 회전 정보: {self.rotation}도")

    def get_rotation(self):
        """ffmpeg로 회전 메타데이터 읽기"""
        try:
            metadata = ffmpeg.probe(self.video_path)
            video_stream = next(
                stream for stream in metadata["streams"] if stream["codec_type"] == "video"
            )
            tags = video_stream.get("tags", {})
            rotation = tags.get("rotate", "0")
            return int(rotation)
        except Exception as e:
            print(f"[WARN] 회전 정보 읽기 실패: {e}")
            return 0

    def rotate_frame(self, frame):
        """회전 값에 따라 프레임 회전"""
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return frame


    def is_portrait(self):
        width, height = self.output_size
        return height > width

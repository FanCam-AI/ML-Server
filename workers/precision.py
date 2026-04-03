from processing import download_saved_models_from_r2, download_dino_v2_base_from_r2
import redis
from ml_service import FaceRecognition,FaceDetection
import boto3
from cryptography.fernet import Fernet
from config import settings
import ray
from .core import process_core
from pathlib import Path


@ray.remote(num_cpus=0.5, num_gpus=0.25, max_concurrency=10)
class PrecisionProcessor:
    def __init__(self):
        self.f = Fernet(settings.FERNET_KEY)
        self.busy = False

        self.redis_client = redis.Redis(
            host=settings.REDIS_CLOUD_HOST,
            port=19268,
            decode_responses=True,
            username="default",
            password=settings.REDIS_CLOUD_PASSWORD,
        )

        self.r2_client = boto3.client(
            "s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY,
            aws_secret_access_key=settings.R2_SECRET_KEY,
            region_name="auto",
        )
        self.base_dir = Path(__file__).parent.parent  # workers/ 상위 = ML-Server

        download_saved_models_from_r2(
            r2_client=self.r2_client,
            bucket_name=settings.R2_BUCKET_NAME,
            download_path="tracking/saved_models"
        )

        download_dino_v2_base_from_r2(
            r2_client=self.r2_client,
            bucket_name=settings.R2_BUCKET_NAME,
            download_path="ml_service/dinov2-base"
        )

        self.person_detection = FaceDetection(
            detection_model_path=str((self.base_dir / 'tracking/saved_models/person_ckpt_best.pth').resolve()),
            detection_model_name="person"
        )

        self.animal_detection = FaceDetection(
                    detection_model_path=str((self.base_dir / 'tracking/saved_models/animal_ckpt_best.pth').resolve()),
                    detection_model_name="animal"
                )

        self.face_recognition = FaceRecognition()

        from tracking import OSTrackTracker
        self.tracker = OSTrackTracker()

    def is_available(self):
        return not self.busy

    def process(self, data):
        self.busy = True
        try:
            detection_model_name = data.get("detection_model_name")
            face_detection = None
            if detection_model_name == "person":
                face_detection = self.person_detection

            elif detection_model_name == "animal":
                face_detection= self.animal_detection

            process_core(
                data,
                redis_client=self.redis_client,
                r2_client=self.r2_client,
                tracker=self.tracker,
                face_detection=face_detection,
                face_recognition=self.face_recognition,
                fernet=self.f
            )

        finally:
            self.busy = False

from processing import download_saved_models_from_r2, download_dino_v2_base_from_r2
import redis
from ml_service import FaceRecognition,FaceDetection
import boto3
from cryptography.fernet import Fernet
from config import settings
import ray
from .core import process_core
from pathlib import Path
from infra import r2_client, redis_client

@ray.remote(num_cpus=1, num_gpus=0.25, max_concurrency=10)
class PrecisionProcessor:
    def __init__(self):
        self.f = Fernet(settings.FERNET_KEY)
        self.busy = False
        self.base_dir = Path(__file__).parent.parent  # workers/ 상위 = ML-Server

        download_saved_models_from_r2(
            r2_client=r2_client,
            bucket_name=settings.R2_BUCKET_NAME,
            download_path="tracking/saved_models"
        )

        download_dino_v2_base_from_r2(
            r2_client=r2_client,
            bucket_name=settings.R2_BUCKET_NAME,
            download_path="ml_service/dinov2-base"
        )

        self.face_detection = FaceDetection(
            detection_model_path=str((self.base_dir / 'tracking/saved_models/person_ckpt_best.pth').resolve()),
            detection_model_name="person"
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
                face_detection = self.face_detection

            process_core(
                data,
                redis_client=redis_client,
                r2_client=r2_client,
                tracker=self.tracker,
                face_detection=face_detection,
                face_recognition=self.face_recognition,
                fernet=self.f
            )

        finally:
            self.busy = False

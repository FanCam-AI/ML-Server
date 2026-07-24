import ray
import redis
import boto3
from cryptography.fernet import Fernet
from config import settings
from .core import process_core

@ray.remote(num_cpus=1, max_concurrency=10)
class NormalProcessor:
    def __init__(self):
        self.f = Fernet(settings.FERNET_KEY)
        self.busy = False


        self.redis_client = redis.Redis(
            host=settings.REDIS_CLOUD_HOST,
            port=settings.REDIS_CLOUD_PORT,
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

    def is_available(self):
        return not self.busy

    def process(self, data):
        self.busy = True
        try:
            process_core(
                data=data,
                redis_client=self.redis_client,
                r2_client=self.r2_client,
                tracker=None,
                face_detection=None,
                face_recognition=None,
                fernet=self.f
            )
        finally:
            self.busy = False

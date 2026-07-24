import ray
import redis
import boto3
from cryptography.fernet import Fernet
from config import settings
from .core import process_core
from infra import r2_client, redis_client

@ray.remote(num_cpus=1, max_concurrency=10)
class NormalProcessor:
    def __init__(self):
        self.f = Fernet(settings.FERNET_KEY)
        self.busy = False

    def is_available(self):
        return not self.busy

    def process(self, data):
        self.busy = True
        try:
            process_core(
                data=data,
                redis_client=redis_client,
                r2_client=r2_client,
                tracker=None,
                face_detection=None,
                face_recognition=None,
                fernet=self.f
            )
        finally:
            self.busy = False

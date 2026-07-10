from google.cloud import secretmanager

PROJECT_ID = "fancam-ai"


def get_secret(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("utf-8")

class Settings:
    def __init__(self):
        self.R2_ACCESS_KEY = get_secret("R2_ACCESS_KEY")
        self.R2_SECRET_KEY = get_secret("R2_SECRET_KEY")
        self.R2_BUCKET_NAME = get_secret("R2_BUCKET")
        self.R2_ENDPOINT_URL = get_secret("R2_ENDPOINT")

        self.REDIS_CLOUD_HOST = get_secret("REDIS_CLOUD_HOST")
        self.REDIS_CLOUD_PASSWORD = get_secret("REDIS_CLOUD_PASSWORD")

        self.FERNET_KEY = get_secret("FERNET_KEY")
        self.API_KEY = get_secret("RUNPOD_API_KEY")
        self.NORMAL_WORKER_COUNT = 5
        self.PRECISION_WORKER_COUNT = 2
        self.SERVERLESS_ENVIRONMENT = False


        self.PORT = 80
        self.PORT_HEALTH = 80


settings = Settings()
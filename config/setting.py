from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    R2_ACCESS_KEY: str
    R2_SECRET_KEY: str
    R2_BUCKET_NAME: str
    R2_ENDPOINT_URL: str
    REDIS_CLOUD_HOST: str
    REDIS_CLOUD_PASSWORD: str
    FERNET_KEY: str
    PORT: int
    PORT_HEALTH: int
    REDIS_CLOUD_PORT: int
    NORMAL_WORKER_COUNT:int = 5
    PRECISION_WORKER_COUNT:int = 4
    API_KEY:str
    SERVERLESS_ENVIRONMENT:bool = True


    # pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()
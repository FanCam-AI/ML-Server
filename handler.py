from processing import MakeResult, cleanup_temp_files, download_r2_keys_to_temp, cleanup_r2_objects, call_save_result_api, download_saved_models_from_r2, download_dino_v2_base_from_r2, call_get_current_user_id_api
import redis
from tracking import Tracking
from ml_service import FaceRecognition,FaceDetection
import boto3
from cryptography.fernet import Fernet
import runpod
from config import settings
import torch


def process_result(data):
    temp_paths = None
    face_detection = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_key = data.get("video_key")
    target_image_keys = data.get("target_image_keys")
    spot_list = data.get("spot_list")
    video_or_gif = data.get("video_or_gif")
    detection_model_name = data.get("detection_model_name")
    drag_box = data.get("drag_box")
    tracking_mode = data.get("tracking_mode")
    encrypted_token = data.get("encrypted_token")
    f = Fernet(settings.FERNET_KEY)
    user_token = f.decrypt(encrypted_token.encode()).decode()
    current_user_id = call_get_current_user_id_api(user_token=user_token)

    redis_client = redis.Redis(
        host=settings.REDIS_CLOUD_HOST,
        port=settings.REDIS_CLOUD_PORT,
        decode_responses=True,
        username="default",
        password=settings.REDIS_CLOUD_PASSWORD,
    )

    r2_client = boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY,
        aws_secret_access_key=settings.R2_SECRET_KEY,
        region_name="auto",
    )

    try:
        tracking = None
        temp_paths = download_r2_keys_to_temp(
            r2_client=r2_client,
            bucket_name=settings.R2_BUCKET_NAME,
            video_key=video_key,
            target_image_keys=target_image_keys
        )
        video_path = temp_paths["video_path"]
        target_image_paths = temp_paths["image_paths"]


        if settings.SERVERLESS_ENVIRONMENT:
            if device == "cpu":
                tracking = Tracking(
                    tracker=None,
                    video_path=video_path,
                    query_image_paths=target_image_paths,
                    face_detection=None,
                    face_recognition=None,
                    detection_model_name=detection_model_name,
                    redis_client=redis_client
                )
            elif device == "cuda":
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

                from tracking import OSTrackTracker

                if detection_model_name == "person":
                    face_detection = FaceDetection(
                        detection_model_path='tracking/saved_models/person_ckpt_best.pth',
                        detection_model_name=detection_model_name
                    )
                elif detection_model_name == "animal":
                    face_detection = FaceDetection(
                        detection_model_path='tracking/saved_models/animal_ckpt_best.pth',
                        detection_model_name=detection_model_name
                    )
                face_recognition = FaceRecognition()

                tracking = Tracking(
                    tracker=OSTrackTracker(),
                    video_path=video_path,
                    query_image_paths=target_image_paths,
                    face_detection=face_detection,
                    face_recognition=face_recognition,
                    detection_model_name=detection_model_name,
                    redis_client=redis_client
                )



        make_result = MakeResult(tracking, video_path, spot_list, current_user_id, r2_client, settings.R2_BUCKET_NAME, tracking_mode, drag_box, redis_client)

        if video_or_gif == "video":
            output_path_list = make_result.make_video()
            file_type = "video/mp4"
        elif video_or_gif == "gif":
            output_path_list = make_result.make_gif()
            file_type = "image/gif"
        else:
            raise Exception("Invalid video_or_gif option")

        for output_path in output_path_list:
            call_save_result_api(output_path=output_path, file_type=file_type, user_token=user_token)

        redis_client.set(f"job_status:{current_user_id}", "done", ex=3600)
        redis_client.set(f"job_progress:{current_user_id}", 100, ex=3600)


    except Exception:
        redis_client.set(f"job_status:{current_user_id}", "error", ex=3600)
        raise

    finally:
        cleanup_temp_files(temp_paths)
        cleanup_r2_objects(
            r2_client,
            settings.R2_BUCKET_NAME,
            video_key,
            target_image_keys
        )


runpod.serverless.start({"handler": process_result})
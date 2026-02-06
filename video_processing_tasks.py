from processing import MakeResult, cleanup_temp_files, download_r2_keys_to_temp, cleanup_r2_objects, call_save_result_api
import redis
from tracking import Tracking, OSTrackTracker
from ml_service import FaceRecognition,FaceDetection
import boto3
import os


def process_result(event):
    temp_paths = None
    face_detection = None
    data = event.get("input", {})
    video_key = data.get("video_key")
    target_image_keys = data.get("target_image_keys")
    spot_list = data.get("spot_list")
    video_or_gif = data.get("video_or_gif")
    detection_model_name = data.get("detection_model_name")
    current_user_id = data.get("current_user_id")
    user_token = data.get("user_token")
    r2_access_key= os.environ.get("R2_ACCESS_KEY")
    r2_secret_key = os.environ.get("R2_SECRET_KEY")
    r2_bucket_name = os.environ.get("R2_BUCKET_NAME")
    redis_cloud_host = os.environ.get("REDIS_CLOUD_HOST")
    redis_cloud_password = os.environ.get("REDIS_CLOUD_PASSWORD")


    # required_fields = [
    #     "video_key", "target_image_keys", "spot_list", "video_or_gif",
    #     "detection_model_name", "current_user_id",
    #     "r2_access_key", "r2_secret_key", "bucket_name", "user_token"
    # ]
    #
    # missing = [f for f in required_fields if not data.get(f)]
    # if missing:
    #     pass

    redis_client = redis.Redis(
        host=redis_cloud_host,
        port=19268,
        decode_responses=True,
        username="default",
        password=redis_cloud_password,
    )

    r2_client = boto3.client(
        "s3",
        endpoint_url="https://9e72ad6e1ccbc2422d24710d0f840b83.r2.cloudflarestorage.com",
        aws_access_key_id=r2_access_key,
        aws_secret_access_key=r2_secret_key,
        region_name="auto",
    )


    try:
        temp_paths = download_r2_keys_to_temp(
            r2_client=r2_client,
            bucket_name=r2_bucket_name,
            video_key=video_key,
            target_image_keys=target_image_keys
        )
        video_path = temp_paths["video_path"]
        target_image_paths = temp_paths["image_paths"]

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

        make_result = MakeResult(tracking, video_path, spot_list, current_user_id, r2_client, r2_bucket_name, redis_client)

        if video_or_gif == "video":
            output_path_list = make_result.make_video()
            file_type = "result/video/mp4"
        elif video_or_gif == "gif":
            output_path_list = make_result.make_gif()
            file_type = "image/gif"
        else:
            raise Exception("Invalid video_or_gif option")

        for output_path in output_path_list:
            call_save_result_api(output_path=output_path, file_type=file_type, user_token=user_token)

        redis_client.set(f"job_status:{current_user_id}", "done", ex=3600)
        redis_client.set(f"job_progress:{current_user_id}", 100, ex=3600)


    except Exception as e:
        redis_client.set(f"job_status:{current_user_id}", "error", ex=3600)
        raise e

    finally:
        cleanup_temp_files(temp_paths)
        cleanup_r2_objects(
            r2_client,
            r2_bucket_name,
            video_key,
            target_image_keys
        )


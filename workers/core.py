from processing import MakeResult, cleanup_temp_files, download_r2_keys_to_temp, cleanup_r2_objects, call_save_result_api, download_saved_models_from_r2, download_dino_v2_base_from_r2, call_get_current_user_id_api
from tracking import Tracking
from config import settings


def process_core(
    data,
    redis_client,
    r2_client,
    tracker=None,
    face_detection=None,
    face_recognition=None,
    fernet=None
):
    temp_paths = None

    video_key = data.get("video_key")
    target_image_keys = data.get("target_image_keys")
    spot_list = data.get("spot_list")
    video_or_gif = data.get("video_or_gif")
    detection_model_name = data.get("detection_model_name")
    drag_box = data.get("drag_box")
    tracking_mode = data.get("tracking_mode")
    encrypted_token = data.get("encrypted_token")
    user_token = fernet.decrypt(encrypted_token.encode()).decode()
    current_user_id = call_get_current_user_id_api(user_token=user_token)

    try:
        temp_paths = download_r2_keys_to_temp(
            r2_client=r2_client,
            bucket_name=settings.R2_BUCKET_NAME,
            video_key=video_key,
            target_image_keys=target_image_keys
        )

        video_path = temp_paths["video_path"]
        target_image_paths = temp_paths["image_paths"]

        tracking = Tracking(
            tracker=tracker,
            video_path=video_path,
            query_image_paths=target_image_paths,
            face_detection=face_detection,
            face_recognition=face_recognition,
            detection_model_name=detection_model_name,
            redis_client=redis_client
        )

        make_result = MakeResult(tracking, video_path, spot_list, current_user_id, r2_client, settings.R2_BUCKET_NAME,
                                 tracking_mode, drag_box, redis_client)

        if video_or_gif == "video":
            output_path_list = make_result.make_video()
            file_type = "video/mp4"
        elif video_or_gif == "gif":
            output_path_list = make_result.make_gif()
            file_type = "image/gif"
        else:
            raise Exception("Invalid video_or_gif option")

        for output_path in output_path_list:
            call_save_result_api(
                output_path=output_path,
                file_type=file_type,
                user_token=user_token
            )

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
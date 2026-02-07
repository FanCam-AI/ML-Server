import os
import tempfile

def download_r2_keys_to_temp(
    r2_client,
    bucket_name,
    video_key: str,
    target_image_keys: list
):
    """
    r2_client: boto3 S3 client (R2)
    bucket_name: R2 bucket name
    video_key: str
    target_image_keys: list[str]
    """

    temp_paths = {
        "video_path": None,
        "image_paths": []
    }

    # ---------- video key ----------
    if video_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            r2_client.download_fileobj(bucket_name, video_key, temp_video)
            temp_paths["video_path"] = temp_video.name

    # ---------- target image keys ----------
    if target_image_keys:
        for key in target_image_keys:
            if not key:
                continue

            _, ext = os.path.splitext(key)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_img:
                r2_client.download_fileobj(bucket_name, key, temp_img)
                temp_paths["image_paths"].append(temp_img.name)

    return temp_paths


def cleanup_temp_files(temp_paths: dict):
    # video
    video_path = temp_paths.get("video_path")
    if video_path and os.path.exists(video_path):
        os.remove(video_path)

    # images
    for img_path in temp_paths.get("image_paths", []):
        if img_path and os.path.exists(img_path):
            os.remove(img_path)


def cleanup_r2_objects(r2_client, bucket_name, video_key, target_image_keys):
    keys = []

    if video_key:
        keys.append(video_key)

    if target_image_keys:
        keys.extend([k for k in target_image_keys if k])

    if not keys:
        return

    # 🔥 실제 R2 삭제
    r2_client.delete_objects(
        Bucket=bucket_name,
        Delete={
            "Objects": [{"Key": k} for k in keys],
            "Quiet": True
        }
    )



def upload_to_r2(r2_client, bucket: str, local_path: str, object_name: str):
    r2_client.upload_file(
        local_path,
        bucket,
        object_name,
        ExtraArgs={
            "ContentType": "image/gif",
            "ACL": "public-read"   # 필요 없으면 제거
        }
    )

    return object_name


def upload_saved_model_to_r2(r2_client, bucket: str, local_path: str, object_name: str):
    r2_client.upload_file(
        local_path,
        bucket,
        object_name
    )



def download_saved_models_from_r2(r2_client, bucket_name: str, download_path: str):
    r2_saved_model_keys = [
        "ml_saved_models/animal_ckpt_best.pth",
        "ml_saved_models/person_ckpt_best.pth",
        "ml_saved_models/vitb_256_mae_ce_32x4_ep300.pth"
    ]

    os.makedirs(download_path, exist_ok=True)

    for key in r2_saved_model_keys:
        filename = os.path.basename(key)  # animal_ckpt_best.pth
        local_path = os.path.join(download_path, filename)

        if os.path.exists(local_path):
            continue

        with open(local_path, "wb") as f:
            r2_client.download_fileobj(bucket_name, key, f)
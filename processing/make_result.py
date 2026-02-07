from .apply_audio import audio_clip_to_video
from .convert_to_gif import convert_to_gif
from .r2_service import upload_to_r2
import os
import secrets
from pathlib import Path
from contextlib import suppress
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_DIR = os.path.join(BASE_DIR, "result", "video")
os.makedirs(VIDEO_DIR, exist_ok=True)
GIF_DIR = os.path.join(BASE_DIR, "result", "gif")
os.makedirs(GIF_DIR, exist_ok=True)



class MakeResult:
    def __init__(self,tracking, video_path, spot_list, user_id, r2_client, bucket_name, redis_client):
        self.tracking = tracking
        self.spot_list = spot_list
        self.video_path = video_path
        self.user_id = user_id
        self.r2_client = r2_client
        self.bucket_name = bucket_name
        self.redis_client = redis_client

    @staticmethod
    def secure_filename():
        unique_filename = secrets.token_urlsafe(16)
        return unique_filename


    @staticmethod
    def remove_if_exists(path_str: str) -> None:
        p = Path(path_str)

        with suppress(FileNotFoundError, PermissionError):
            if p.exists():
                p.unlink()


    def make_gif(self):
        output_path_list = list()
        for i in range(len(self.spot_list)):
            time = self.spot_list[f'spot_{i}']
            start_time = time[0]
            end_time = time[1]
            start_second = start_time[0] * 3600 + start_time[1] * 60 + start_time[2]
            end_second = end_time[0] * 3600 + end_time[1] * 60 + end_time[2]
            print(start_second, end_second)
            self.tracking.tracking_idol(start_time=start_second, end_time=end_second, user_id=self.user_id)
            base_name = os.path.basename(self.video_path)
            base, ext = os.path.splitext(self.video_path)
            processed_video_path = f"{base}_output.mp4"
            unique_filename = self.secure_filename()
            convert_to_gif(video_path=processed_video_path, output_path=os.path.join(GIF_DIR, f"{unique_filename}.gif"))
            self.redis_client.set(f"job_progress:{self.user_id}", 95, ex=3600)
            output_path = os.path.join(GIF_DIR, f"{unique_filename}.gif")
            r2_object_name = f"result/gif/{unique_filename}.gif"
            upload_to_r2(
                r2_client=self.r2_client,
                bucket=self.bucket_name,
                local_path=output_path,
                object_name=r2_object_name
            )
            output_path_list.append(r2_object_name)
            self.remove_if_exists(processed_video_path)
            self.remove_if_exists(output_path)

        return output_path_list


    def make_video(self):
        output_path_list = list()
        print(self.spot_list)
        for i in range(len(self.spot_list)):
            time = self.spot_list[f'spot_{i}']
            start_time = time[0]
            end_time = time[1]
            start_second = start_time[0] * 3600 + start_time[1] * 60 + start_time[2]
            end_second = end_time[0] * 3600 + end_time[1] * 60 + end_time[2]
            print(start_second, end_second)
            self.tracking.tracking_idol(start_time=start_second, end_time=end_second, user_id= self.user_id,visualize=False)
            base_name = os.path.basename(self.video_path)
            base, ext = os.path.splitext(self.video_path)
            processed_video_path = f"{base}_output.mp4"
            unique_filename = self.secure_filename()
            audio_clip_to_video(original_video_path=self.video_path, processed_video_path=processed_video_path,
                                output_video_path = os.path.join(VIDEO_DIR, f"{unique_filename}.mp4"), start_audio_time=start_second,
                                end_audio_time=end_second)

            self.redis_client.set(f"job_progress:{self.user_id}", 95, ex=3600)
            output_path = os.path.join(VIDEO_DIR, f"{unique_filename}.mp4")
            r2_object_name = f"result/video/{unique_filename}.mp4"
            upload_to_r2(
                r2_client=self.r2_client,
                bucket=self.bucket_name,
                local_path=output_path,
                object_name=r2_object_name
            )
            output_path_list.append(r2_object_name)
            self.remove_if_exists(processed_video_path)
            self.remove_if_exists(output_path)

        return output_path_list
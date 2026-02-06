import os
from moviepy.editor import VideoFileClip


def audio_clip_to_video(original_video_path, processed_video_path, output_video_path, start_audio_time, end_audio_time):
    #
    # temp_audio_path = os.path.join("static", "temp", "temp-audio.m4a")
    # os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)

    # 원본 비디오에서 오디오 추출
    original_video_clip = VideoFileClip(original_video_path)
    audio_clip_duration = original_video_clip.duration

    # start_time, end_time 범위 보정
    if start_audio_time >= audio_clip_duration:

        start_audio_time = max(0, audio_clip_duration - 1)  # 마지막 0.1초라도 남기기
    if end_audio_time > audio_clip_duration:
        end_audio_time = audio_clip_duration

    if end_audio_time > audio_clip_duration:
        end_audio_time = audio_clip_duration
    audio_clip = original_video_clip.audio.subclip(start_audio_time, end_audio_time)

    processed_video_clip = VideoFileClip(processed_video_path)
    processed_video_clip = processed_video_clip.set_audio(audio_clip)

    # 최종 영상 저장 (임시 오디오 파일 지정 + 자동 삭제)
    processed_video_clip.write_videofile(
        output_video_path,
        codec='libx264',
        audio_codec='aac',
        bitrate='30000k',
        remove_temp=True
    )

    # 리소스 정리
    processed_video_clip.close()
    original_video_clip.close()

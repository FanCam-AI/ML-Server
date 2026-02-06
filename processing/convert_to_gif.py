import subprocess
import os

def convert_to_gif(video_path, output_path):
    palette = "palette.png"

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "fps=30,scale=480:-1:flags=lanczos,palettegen",
        palette
    ])
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", palette,
        "-filter_complex", "fps=30,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse",
        output_path
    ])

    # 변환 끝나고 팔레트 삭제
    if os.path.exists(palette):
        os.remove(palette)
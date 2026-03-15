import subprocess
import tempfile

def convert_webm_to_mp4(webm_path):
    mp4_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", webm_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return mp4_path
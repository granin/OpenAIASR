import argparse
import io
import time
import os
from functools import wraps
from pathlib import Path
from typing import Optional, List
import av
import ffmpy
import openai
import requests
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import subprocess


AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.m4a', '.mp4', '.mpeg']


class APIError(Exception):
    pass


def retry(tries: int = 3, delay: float = 5.0, **kwargs):
    def decorator_retry(f):
        @wraps(f)
        def f_retry(*args, **f_kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs, **f_kwargs)
                except Exception as e:
                    msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                    print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= 2
            return f(*args, **kwargs, **f_kwargs)
        return f_retry
    return decorator_retry


def is_valid_audio_video(file_path):
    try:
        metadata = ffmpy.FFprobe(inputs={file_path: None}).run(
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except ffmpy.FFExecutableNotFoundError:
        print(
            "FFmpeg not found. Please install it and ensure it's in your system's PATH.")
        sys.exit(1)
    except ffmpy.FFExecutableNotFoundError:
        return False

    return True


def get_file_size(file_path: Path) -> int:
    return file_path.stat().st_size


def convert_bytes_to_mb(bytes: int) -> float:
    return bytes / (1024 * 1024)


@retry(tries=3, delay=5.0)
def transcribe_audio(file_path: str, print_transcript: bool = False) -> Optional[dict]:
    if print_transcript:
        print(f"Transcribing:\n{file_path}\n")

    file_size = get_file_size(Path(file_path))

    max_file_size = 26214400  # 25 MB
    if file_size > max_file_size:
        print(
            f"Error: {file_path} exceeds the API file size limit of {max_file_size} bytes.")
        return None

    with open(file_path, "rb") as audio_file:
        file_size_mb = convert_bytes_to_mb(file_size)
        print(f"Before API call for:\n{file_path}\n")
        print(f"{file_size_mb:.2f} MB")
        start_time = time.time()

        try:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file
            )

        except APIError as e:
            print(
                f"Error during transcription API call for {file_path}: {e}")
            return None

        end_time = time.time()

        print(f"After API call for:\n{file_path}\n")

        time_taken = end_time - start_time
        minutes, seconds = divmod(time_taken, 60)

        print(
            f"API call for {file_path} took {int(minutes)} minutes and {int(seconds)} seconds")

        if transcript is None:
            print(f"No transcript returned for {file_path}")
            return None

        if print_transcript:
            print(transcript['text'])
        return transcript


def generate_txt(file_path: str, print_transcript: bool = False) -> Optional[str]:
    try:
        transcript = transcribe_audio(file_path, print_transcript=True)
        if transcript is None:
            print(f"Error processing {file_path}: No transcript returned")
            return None
        transcript_text = transcript['text']
        return transcript_text
    except APIError as e:
        print(f"Error processing {file_path}: {e}")
        return None


def get_subtitles(file: str, subtitle_format: str = 'srt', timeout: int = 600, skip_vtt: bool = False, tries: int = 3, delay: float = 5.0, **kwargs) -> Optional[str]:
    if skip_vtt and subtitle_format == 'vtt':
        return None
    try:
        url = 'https://api.openai.com/v1/audio/transcriptions'
        headers = {
            'Authorization': f'Bearer {openai.api_key}',
        }
        data = {
            'model': 'whisper-1',
            'response_format': subtitle_format,
            'language': 'en',
        }
        data.update(kwargs)
        files = {
            'file': (str(file), open(str(file), 'rb'))
        }

        response = requests.post(url, headers=headers,
                                 files=files, data=data, timeout=timeout)

        if response.status_code == 200:
            return response.text
        else:
            print(
                f"Error generating {subtitle_format.upper()} for {file}: {response.text}")
            return None
    except APIError as e:
        print(f"Error generating {subtitle_format.upper()} for {file}: {e}")
        return None


def generate_vtt(file_path: str) -> Optional[str]:
    return get_subtitles(file_path, subtitle_format='vtt', tries=args.api_retries, delay=args.api_retry_delay)


def generate_srt_from_vtt(vtt_content: str) -> str:
    return convert_vtt_to_srt(vtt_content)


def save_subtitle(file_path: str, content: str, fmt: str) -> None:
    if content is not None:
        with open(os.path.splitext(file_path)[0] + f".{fmt}", 'w', encoding='utf-8') as f:
            f.write(content)


def is_audio_video(file: Path) -> bool:
    return file.suffix.lower() in AUDIO_EXTENSIONS


def transcription_files_exist(root: Path, file_name: str, overwrite: str) -> bool:
    txt_path = root / f"{file_name}.txt"
    vtt_path = root / f"{file_name}.vtt"
    srt_path = root / f"{file_name}.srt"

    if overwrite == "none":
        return txt_path.is_file() or vtt_path.is_file() or srt_path.is_file()
    elif overwrite == "one":
        return txt_path.is_file() and vtt_path.is_file() and srt_path.is_file()
    else:
        return False


def convert_vtt_to_srt(vtt_content: str) -> str:
    vtt_lines = vtt_content.splitlines()
    srt_file = io.StringIO()
    index = 1

    for line in vtt_lines:
        if line.strip() == '':
            srt_file.write('\n')
            index += 1
        elif len(line.strip().split()) == 1 and line.strip().split()[0].isdigit():
            srt_file.write(str(index) + '\n')
        else:
            srt_file.write(line + '\n')

    return srt_file.getvalue()


def transcribe_and_save(file_path: str, formats: List[str], print_transcript: bool = False) -> None:
    txt_content, vtt_content, srt_content = None, None, None

    if 'txt' in formats:
        txt_content = generate_txt(file_path, print_transcript)
        save_subtitle(file_path, txt_content, 'txt')
        file_size = get_file_size(file_path)

    if 'vtt' in formats:
        vtt_content = generate_vtt(file_path)
        save_subtitle(file_path, vtt_content, 'vtt')

    if 'srt' in formats:
        if vtt_content is None:
            vtt_content = generate_vtt(file_path)
        srt_content = generate_srt_from_vtt(vtt_content)
        save_subtitle(file_path, srt_content, 'srt')

    input("Transcription complete. Press Enter to continue...")


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    input_container = av.open(video_path)
    audio_stream = None

    for stream in input_container.streams:
        if stream.type == "audio":
            audio_stream = stream
            break

    if not audio_stream:
        raise Exception("No audio stream found in the video file")

    output_container = av.open(audio_path, "w")
    output_stream = output_container.add_stream(template=audio_stream)

    for packet in input_container.demux(audio_stream):
        for frame in packet.decode():
            output_container.mux(frame)

    output_container.close()


def process_files(folder_path: str, formats: List[str], overwrite: str, print_transcript: bool = False) -> None:
    processed_files = []

    # Get the total number of files to process
    total_files = sum(1 for _ in Path(folder_path).rglob('*')
                      if _.suffix.lower() in AUDIO_EXTENSIONS)
    progress_bar = tqdm(total=total_files, desc="Processing files", ncols=100)

    for root, _, files in os.walk(folder_path):
        root = Path(root)
        for file in files:
            if file.endswith(tuple(AUDIO_EXTENSIONS)):
                file_path = root / file

                # Validate the audio or video file
                if not is_valid_audio_video(file_path):
                    print(f"Skipping {file_path}:")
                    print("File already processed in previous run")
                    continue

                # Check if txt, vtt, or srt files exist
                txt_exists = os.path.isfile(
                    os.path.splitext(file_path)[0] + ".txt")
                vtt_exists = os.path.isfile(
                    os.path.splitext(file_path)[0] + ".vtt")
                srt_exists = os.path.isfile(
                    os.path.splitext(file_path)[0] + ".srt")

                # Skip if the files exist
                if overwrite == "none" and (txt_exists or vtt_exists or srt_exists):
                    print(f"Skipping {file_path}:")
                    print("TXT, VTT, or SRT files already exist")
                    continue
                elif overwrite == "one" and (txt_exists and vtt_exists and srt_exists):
                    print(f"Skipping {file_path}:")
                    print("TXT, VTT, or SRT files already exist")
                    continue
                elif overwrite == "all":
                    pass

                if file_path in processed_files:
                    print(f"Skipping {file_path}:")
                    print("File already processed in previous run")
                    continue

                if transcription_files_exist(root, file_path.stem, overwrite):
                    print(f"Skipping {file_path}:")
                    print("TXT, VTT, or SRT files already exist")
                    continue

                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)
                if file_size > 26214400:
                    print(
                        f"Skipping {file_path}: file size is too large ({file_size_mb:.2f} MB)")
                    continue

                if file.lower().endswith(('.mp4', '.mpeg')):
                    audio_file_path = os.path.splitext(file_path)[0] + ".mp3"
                    if not os.path.exists(audio_file_path):
                        try:
                            extract_audio_from_video(
                                file_path, audio_file_path)
                        except Exception as e:
                            print(
                                f"Error extracting audio from {file_path}: {e}")
                            continue
                    try:
                        transcribe_and_save(
                            file_path, formats, print_transcript)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
                    transcribe_and_save(
                        audio_file_path, formats, print_transcript)
                else:
                    transcribe_and_save(file_path, formats, print_transcript)

                processed_files.append(file_path)
                progress_bar.update(1)
    progress_bar.close()


def load_api_key(file_path: str) -> Optional[str]:

    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]

    api_key_path = Path(file_path)
    if api_key_path.is_file():
        with open(file_path, "r") as f:
            return f.read().strip()

    return None


def save_api_key(file_path: str, api_key: str) -> None:
    with open(file_path, "w") as f:
        f.write(api_key)


parser = argparse.ArgumentParser(
    description="Transcribe audio and video files using OpenAI's Whisper ASR API.")

input_group = parser.add_argument_group('Input')
input_group.add_argument("folder_path",
                         help="The path to the folder containing the audio and video files to be transcribed.")

output_group = parser.add_argument_group('Output')
output_group.add_argument("-f", "--formats", nargs="+", choices=["txt", "vtt", "srt"], default=["txt", "vtt", "srt"],
                          help="The subtitle formats to generate. Multiple formats can be specified. Available choices are: txt, vtt, srt. Default: txt, vtt, srt.")
output_group.add_argument("-o", "--overwrite", choices=["none", "one", "all"], default="none",
                          help="Overwrite behavior: 'none' to skip if any file exists, 'one' to skip if all files exist, 'all' to overwrite files. Default: 'none'.")

api_group = parser.add_argument_group('API')
api_group.add_argument("-k", "--api-key-file", type=str, default="openai_api_key.txt",
                       help="Path to the file containing your OpenAI API key. Default: 'openai_api_key.txt'.")
api_group.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds for the OpenAI API requests. Default: 600.")
api_group.add_argument("--api-retries", type=int, default=3,
                       help="Number of retries for the OpenAI API requests. Default: 3.")
api_group.add_argument("--api-retry-delay", type=float, default=5.0,
                       help="Delay in seconds between retries for the OpenAI API requests. Default: 5.0.")

misc_group = parser.add_argument_group('Miscellaneous')
misc_group.add_argument("-t", "--transcript", action="store_true",
                        help="Print the transcript generated by OpenAI API.")

args = parser.parse_args()
if __name__ == "__main__":

    api_key = load_api_key(args.api_key_file)
    if api_key is None:
        api_key = input("Enter your OpenAI API key: ").strip()
        save_api_key(args.api_key_file, api_key)

    openai.api_key = api_key
    requests.adapters.DEFAULT_POOL_TIMEOUT = args.timeout

    process_files(args.folder_path, args.formats, args.overwrite,
                  print_transcript=args.transcript)

import yt_dlp
import os
import argparse

def download_audio(url, output_dir="downloads"):
    """
    Download audio from a YouTube URL and save it as MP3.
    
    Args:
        url (str): YouTube URL
        output_dir (str): Directory to save the audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
    }
    
    # Download the audio
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from: {url}")
            info = ydl.extract_info(url, download=True)
            print(f"Successfully downloaded: {info['title']}")
            return os.path.join(output_dir, f"{info['title']}.mp3")
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download audio from YouTube URL')
    parser.add_argument('url', help='YouTube URL to download')
    parser.add_argument('--output-dir', default='downloads', help='Directory to save the audio file')
    
    args = parser.parse_args()
    
    output_file = download_audio(args.url, args.output_dir)
    if output_file:
        print(f"Audio saved to: {output_file}")

if __name__ == "__main__":
    main()

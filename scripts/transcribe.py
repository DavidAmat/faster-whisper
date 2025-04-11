import sys
import os
import argparse
from pathlib import Path
from faster_whisper import WhisperModel

def get_base_filename(filepath):
    """Extract base filename without extension."""
    return Path(filepath).stem

def transcribe_audio(audio_path, output_dir=None, language=None):
    """
    Transcribe audio file and save output to a text file.
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str, optional): Directory to save the transcript. If None, uses default location.
    """
    # Set default output directory if not specified
    if output_dir is None:
        # Get the parent directory of audio_files (which is the audios directory)
        audio_files_dir = os.path.dirname(audio_path)
        audios_dir = os.path.dirname(audio_files_dir)
        output_dir = os.path.join(audios_dir, "audio_transcripts")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = get_base_filename(audio_path)
    output_file = os.path.join(output_dir, f"{base_filename}.txt")
    
    # Initialize the model
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    # Transcribe the audio
    segments, info = model.transcribe(audio_path, beam_size=5, language=language)
    
    # Write the transcript to file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header with language information
        f.write(f"Detected language: {info.language} (probability: {info.language_probability:.2f})\n\n")
        
        # Write segments
        for segment in segments:
            f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
    
    print(f"Transcript saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio file using Whisper')
    parser.add_argument('audio_file', help='Path to the audio file')
    parser.add_argument('--output-dir', help='Directory to save the transcript (optional)')
    parser.add_argument('--language', help='Language code for transcription (optional)')  # New argument for language

    args = parser.parse_args()
    
    # If no full path is provided, assume it's in the default audio_files directory
    if not os.path.isabs(args.audio_file):
        default_audio_dir = "/home/david/Documents/audios/audio_files"
        args.audio_file = os.path.join(default_audio_dir, args.audio_file)
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    transcribe_audio(args.audio_file, args.output_dir, args.language)

if __name__ == "__main__":
    main()

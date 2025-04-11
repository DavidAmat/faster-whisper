import argparse
import os
import sys
import io
import speech_recognition as sr
from faster_whisper import WhisperModel
from queue import Queue, Empty
from time import sleep
from tempfile import NamedTemporaryFile
from pathlib import Path
from datetime import datetime

# Global flag to signal exit
exit_flag = False

def get_script_directory():
    """Get the directory where the script is located."""
    return Path(__file__).resolve().parent

def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread-safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)

def main():
    global exit_flag # Allow modifying the global flag

    parser = argparse.ArgumentParser(description='Real-time transcription from microphone using Faster Whisper')

    # --- Arguments from original script ---
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the transcript file (default: script directory)')
    parser.add_argument('--output_file_name', type=str, default="live_transcript.txt",
                        help='Name of the output transcript file (default: live_transcript.txt)')

    # --- Arguments from second script (adapted) ---
    parser.add_argument("--model", default="large-v3", help="Faster Whisper model size",
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                                 "medium", "medium.en", "large-v1", "large-v2", "large-v3", "distil-large-v2"])
    parser.add_argument("--device", default="cuda", help="Device for inference ('cuda', 'cpu')",
                        choices=["cuda", "cpu"])
    parser.add_argument("--compute_type", default="float16", help="Compute type for GPU ('float16', 'int8_float16', 'int8') or CPU ('int8', 'float32')",
                        choices=["float16", "int8_float16", "int8", "float32"])
    parser.add_argument("--language", default="en", type=str,
                    help="Language code (e.g., 'en', 'es', 'fr'). Default is English.")
    parser.add_argument("--threads", default=4, help="Number of threads for CPU inference", type=int)

    # --- Microphone specific arguments ---
    parser.add_argument("--energy_threshold", default=500, # Slightly lower default than example 2
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=5.0,  # Changed from 2.0 to 5.0
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=1.5, # How much silence triggers transcription
                        help="Seconds of silence before processing the audio phrase.", type=float)
    parser.add_argument("--list-mics", action='store_true',
                        help="List available microphone devices and exit.")
    parser.add_argument("--mic-index", default=None, type=int, help="Index of the microphone to use.")
    parser.add_argument("--mic-name", default=None, type=str,
                        help="Partial or full name of the microphone to use (e.g., 'pulse', 'default', 'USB PnP Sound Device').")


    args = parser.parse_args()

    # --- List Microphones and Exit ---
    if args.list_mics:
        print("Available microphone devices are:")
        try:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  Index {index}: \"{name}\"")
        except Exception as e:
            print(f"Could not list microphones: {e}")
            print("Ensure PortAudio or necessary audio backend is installed and configured.")
        sys.exit(0)

    # --- Output File Setup ---
    script_dir = get_script_directory()
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"transcript_{timestamp}.txt"
    output_file = output_dir / output_file_name

    # Clear the file or create it if it doesn't exist at the start
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Transcription started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.flush()

    # --- Model Initialization ---
    # print(f"Loading Whisper model '{args.model}' on device '{args.device}'...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    # print("Model loaded successfully.")
    # print(f"Using {mic_info}.")
    # print("Adjusting for ambient noise... Please wait.")
    # print("Ambient noise adjustment complete.")

    # --- Microphone Setup ---
    global data_queue # Make queue accessible in callback
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False # Crucial: disable dynamic threshold

    source = None
    mic_info = "default system microphone"

    try:
        if args.mic_index is not None:
            mic_info = f"microphone index {args.mic_index}"
            source = sr.Microphone(device_index=args.mic_index, sample_rate=16000)
        elif args.mic_name:
            mic_info = f"microphone matching name '{args.mic_name}'"
            available_mics = sr.Microphone.list_microphone_names()
            found_mic_index = None
            for index, name in enumerate(available_mics):
                if args.mic_name.lower() in name.lower():
                    found_mic_index = index
                    mic_info = f"microphone '{name}' (index {index})"
                    break
            if found_mic_index is not None:
                 source = sr.Microphone(device_index=found_mic_index, sample_rate=16000)
            else:
                print(f"Error: Could not find microphone with name containing '{args.mic_name}'.")
                print("Available microphones:")
                for index, name in enumerate(available_mics):
                    print(f"  Index {index}: \"{name}\"")
                sys.exit(1)
        else:
            # Default microphone
             source = sr.Microphone(sample_rate=16000)

        # print(f"Using {mic_info}.")

        with source:
            # print("Adjusting for ambient noise... Please wait.")
            recorder.adjust_for_ambient_noise(source, duration=1.0) # Adjust for 1 second
            # print("Ambient noise adjustment complete.")

    except Exception as e:
        print(f"Error initializing microphone: {e}")
        print("Please ensure you have a working microphone and the necessary permissions.")
        print("Try running with '--list-mics' to see available devices.")
        sys.exit(1)


    # --- Start Background Listening ---
    # phrase_time_limit=record_timeout means the callback is triggered AT LEAST this often
    # timeout=phrase_timeout means silence for this duration also triggers the callback
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)
    print("Listening... Press Ctrl+C to stop.")

    # --- Main Processing Loop ---
    temp_file = NamedTemporaryFile(suffix=".wav", delete=False).name
    print(f"Using temporary file: {temp_file}")

    # Add a timestamp tracker
    start_time = 0

    try:
        while not exit_flag:
            try:
                # Wait until audio data is available from the queue
                audio_data_raw = data_queue.get(block=True, timeout=0.5)

                # Convert raw data to WAV format in memory
                audio_data = sr.AudioData(audio_data_raw, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write WAV data to the temporary file
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Transcribe the temporary audio file
                segments, info = model.transcribe(temp_file, beam_size=5, language=args.language)

                # Write segments with correct timing
                with open(output_file, 'a', encoding='utf-8') as f:
                    # Combine all segments text into one
                    full_text = ' '.join([segment.text.strip() for segment in segments])
                    if full_text.strip():  # Only write if there's actual text
                        end_time = start_time + args.record_timeout
                        line = f"[{start_time}-{end_time}s] {full_text}\n"
                        f.write(line)
                        print(f"[{start_time}-{end_time}s] {full_text}")  # Print with timestamp for monitoring
                        start_time = end_time  # Update start time for next segment
                    f.flush()

                # Sleep for the record_timeout duration
                sleep(args.record_timeout)

            except Empty:
                # No data in queue, continue checking
                continue
            except Exception as e:
                print(f"\nError during transcription loop: {e}")
                sleep(1)  # Avoid rapid error loops

    except KeyboardInterrupt:
        print("\nStopping listener...")
        exit_flag = True # Signal threads if any are waiting

    finally:
        # --- Cleanup ---
        if 'stop_listening' in locals() and stop_listening:
            stop_listening(wait_for_stop=False) # Stop the background thread
            print("Listener stopped.")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Temporary file {temp_file} removed.")
            except OSError as e:
                 print(f"Error removing temporary file {temp_file}: {e}")
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\nTranscription ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        # Only print the output file location
        print(f"\nTranscription saved to: {output_file}")

if __name__ == "__main__":
    # Make queue global for the callback function
    data_queue = None
    main()
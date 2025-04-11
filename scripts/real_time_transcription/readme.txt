sudo apt-get install portaudio19-dev python3-pyaudio flac

pip install faster-whisper speechrecognition pyaudio


conda activate t_whisper
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
python live_transcribe.py --mic-name "pulse" --model small --device cuda --compute_type float16
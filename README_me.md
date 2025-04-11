This is my personal list of commands to set up the project

# Quick Guide


1. From Grabacion > Compartir > Guardar en Archivos 
2. Go to https://www.icloud.com/iclouddrive/ to the Audios folder
3. Download to /home/david/Documents/audios/audio_files your file
4. Transcribe it using the `transcribe.py` script

```bash
whisper

# Optionally specify a custom output directory
python scripts/transcribe.py Theo.m4a --language en

```

# Install the repository

## 1. Create and Activate Conda Environment
```bash
# Create the conda environment with Python 3.10.14
conda create -y --name t_whisper python=3.10.14

# Activate the environment
conda activate t_whisper
```

## 2. Install Base Requirements
```bash
# Install the base requirements from requirements.txt
pip install -r requirements.txt
```

## 3. Install NVIDIA Libraries (for GPU support)
```bash
conda activate t_whisper
# Install NVIDIA libraries for CUDA 12
pip install nvidia-cublas-cu12
pip install "nvidia-cudnn-cu12==9.*"

# Set LD_LIBRARY_PATH for NVIDIA libraries
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

## 4. Install Additional Dependencies
```bash
# Install yt-dlp for YouTube downloads
pip install yt-dlp

# Install PyTorch with CUDA 12.1 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

# Download video audio mp3 from YT

```bash
python scripts/downloadmp3.py "https://www.youtube.com/watch?v=pq5gCfua5FA&ab_channel=DAZNES" --output-dir "audios/"
```

# Audio Transcription Process

## Directory Structure
The audio files and transcripts are organized in the following structure:
```
/home/david/Documents/audios/
├── audio_files/          # Place your audio files here
│   └── prova1.m4a
└── audio_transcripts/    # Transcripts will be saved here
```

## Transcribing Audio Files

### Basic Usage
1. Save the Video in your iPhone to the iCloud > Audios Folder

2. Go to https://www.icloud.com/iclouddrive/ to the Audios folder

3. Download to /home/david/Documents/audios/audio_files your file

5. Transcribe it using the `transcribe.py` script

### Transcribe file

To transcribe an audio file, use the `transcribe.py` script:

```bash
whisper  # alias in zshrc 

# If the audio file is in the default audio_files directory
python scripts/transcribe.py prova2.m4a

# Or specify the full path to the audio file
python scripts/transcribe.py /path/to/your/audio.mp3

# Optionally specify a custom output directory
python scripts/transcribe.py prova1.m4a --output-dir /custom/output/path
```

### Output Format
The transcript will be saved as a .txt file with the same name as the audio file. For example:
- Input: `audio_files/prova1.m4a`
- Output: `audio_transcripts/prova1.txt`

The transcript file will contain:
1. Header with detected language and probability
2. Timestamped segments of the transcription

Example output:
```
Detected language: es (probability: 0.98)

[0.00s -> 9.54s] El Barça es Chesney en la puerta, Gerard, Íñigo, Araujo.
[9.62s -> 17.40s] Tres c...
```

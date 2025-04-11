import sys
import os
import torch
from faster_whisper import WhisperModel

# Print Python executable and working directory
print(f"Using Python from: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# Check CUDA availability
print("\nCUDA/GPU Information:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Try to initialize WhisperModel to check if it works
print("\nTesting WhisperModel initialization:")
try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("✓ WhisperModel initialized successfully with CUDA")
except Exception as e:
    print(f"✗ Error initializing WhisperModel: {str(e)}")

print("\nAll required libraries imported successfully:")
print("✓ sys")
print("✓ os")
print("✓ torch")
print("✓ faster_whisper")

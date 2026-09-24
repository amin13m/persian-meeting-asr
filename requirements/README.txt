Persian ASR Requirements
========================

Each model/engine has its own requirements file.

Current model:

whisper_persian_v4.txt

The Torch/CUDA stack is managed separately because the
project uses a specific CUDA-compatible PyTorch installation.

Current tested environment:

torch       2.12.1+cu126
torchaudio  2.11.0+cu126
torchvision 0.27.1+cu126

Do not reinstall Torch unless the model specifically requires
a different version.
# Automatic Speech Recognition with Custom Models

## Overview
In this repository, we build, train, and evaluate several speech recognition models. The task of Automatic Speech Recognition (ASR) consists of automatically generating a text transcription starting from a raw speech audio file. To tackle this task, we either build and train deep neural models from scratch, or import pre-trained model and fine-tune them on small labelled dataset. Regarding the former, we design our models taking inspiration from SoTA speech recognition models, i.e. Deep Speech 2, Conformer, Jasper. Regarding the latter, we import Wav2Vec2 pre-trained model from Hugging Face Hub and fine-tune it on a fraction of LIBRISPEECH clean slip and FLEURS dataset.

## Results
We evaluate our models on the LibriSpeech benchmark, and provide results both in Word Error Rate (WER) and Character Error Rate (CER). Despite having limited hardware, we are able to achieve decent performances compared to SoTA models.

## Our Custom Models
- Deep Speech Base (ResNetInv + GRUs)
- Deep Speech Attention (ResNetInv + Encoders)
- Jasper Base (5x3)
- Jasper DR (Dense Residual) (5x3)
- Conformer Small
- Wav2Vec2 Base (pre-trained from Hugging Face)

## Install dependencies

Refer to the ufficial PyTorch documentation to install the correct version of CUDA: https://pytorch.org/get-started/locally/. Here we used an Anaconda environment (https://www.anaconda.com/). 

```
conda create -n myenv python=3.12.2
conda activate myenv
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c pytorch torchtext
conda install numpy cffi
pip install pysoundfile
pip install wandb
```
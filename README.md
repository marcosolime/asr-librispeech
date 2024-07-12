# Automatic Speech Recognition with Custom Models

In this work, we build, train, and evaluate several speech recognition models. The task of Automatic Speech Recognition (ASR) consists of automatically generating a text transcription starting from a raw speech audio file. To tackle this task, we either build and train deep neural models from scratch, or import pre-trained model and fine-tune them on small labelled dataset. 

This project was made for the NLP exam of the Master's Degree in Artificial Intelligence, Università di Bologna.<br>
You can read our <a href="ASR_presentation.pptx">presentation</a> and <a href="ASR_report.pdf">paper report</a> to get more detailed information.

### Contributors:
- Marco Solime
- Alessandro Folloni
- Daniele Napolitano
- Gabriele Fossi
  
## Our Custom Models
- **Deep Speech Base** (ResNetInv + GRUs)
- **Deep Speech Attention** (ResNetInv + Encoders)
- **Jasper Base** (5x3)
- **Jasper DR** (Dense Residual) (5x3)
- **Conformer Small**
- **Wav2Vec2 Base** (pre-trained from Hugging Face, then fine-tuned on a fraction of LIBRISPEECH and FLEURS datasets)

## Results
We evaluate our models on the LibriSpeech benchmark, and provide results both in Word Error Rate (WER) and Character Error Rate (CER). We did not implement a language model to correct speling mistakes. 
Despite this and training with limited hardware, we are able to achieve decent performances compared to SoTA models.<br>
The table below shows our results (top three rows) compared to the baseline results, taken from the original papers, in **Word Error Rate (WER)** metrics: <br>
<img src="https://github.com/marcosolime/asr-librispeech/blob/main/results_table.png" width=600>





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

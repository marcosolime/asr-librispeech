import torch
import torch.nn as nn
import torchaudio

class Preprocessing():

    def __init__(self):
        self.urls = {"train-clean-100",
                     "train-clean-360",
                     "train-clean-500",
                     "dev-clean",
                     "dev-other",
                     "test-clean",
                     "test-other"}
        self.dataset_path = "./audioset"

        # Pipelines
        self.train_pipe = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35))
        self.test_pipe = torchaudio.transforms.MelSpectrogram()
    
    def download(self, split: str, download=False):
        if split not in self.urls:
            raise Exception(f"Wrong argument: split must be in {self.urls}")

        audioset = torchaudio.datasets.LIBRISPEECH(self.dataset_path,
                                                   url=split,
                                                   download=download)
        return audioset

    def get_length(self, audioset):
        return len(audioset)

    def print_raw_sample(self, idx, audioset):
        if idx < 0 or idx >= len(audioset):
            raise Exception(f"Out of bounds: idx must lay between 0-{len(audioset)-1}")
    
        waveform, sample_rate, transcript,\
            speaker_id, chapter_id, utterance_id = audioset[idx]

        print(f"+------ Printing info for sample [{idx}] ------+")
        print(f"- Waveform is a Tensor of size {list(waveform.size())}, type={waveform.dtype}")
        print(f"- Sample rate: {sample_rate}")
        print(f"- Transcript: {transcript[:42]}...")
        print(f"- Speaker id: {speaker_id}")
        print(f"- Chapter id: {chapter_id}")
        print(f"- Utterance id: {utterance_id}")
        print("+---------------------------------------+")
    
    def print_loader_info(self, loader):
        print(f"+------ Dataloader length: {len(loader.dataset)} ------+")
        print(f"# Batches: {len(loader)}")
        for batch_sample in loader:
            print("Spectrogram shape:", list(batch_sample[0].shape))
            print("Label shape:", list(batch_sample[1].shape))
            print("Mel length (length of each spectrogram):", batch_sample[2][:6], "...")
            print("Idx length (length of each label):", batch_sample[3][:6], "...")
            break
        print("+------------------------------------+")

    def preprocess(self, audioset, split: str, stride: int, word_model):
        if split not in self.urls:
            raise Exception(f"Wrong argument: split must be in {self.urls}")

        if split[:5] == "train":
            augment_fn = self.train_pipe
        else:
            augment_fn = self.test_pipe

        print(f"Preprocessing {split}...")

        spectrograms = []
        indices = []
        len_spectrograms = []
        len_indices = []
        
        for waveform, _, transcript, _, _, _ in audioset:
            # Augment audio data
            spec = augment_fn(waveform).squeeze(0).transpose(0,1)
            spectrograms.append(spec)

            # Convert text transcript to sequence of ids
            ids = torch.Tensor(word_model.text_to_int(transcript.lower()))
            indices.append(ids)

            # Append audio and text length
            len_spectrograms.append(spec.shape[0]//stride)
            len_indices.append(len(ids))
        
        # Zero pad
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        indices = nn.utils.rnn.pad_sequence(indices, batch_first=True)

        return spectrograms, indices, len_spectrograms, len_indices

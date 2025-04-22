import numpy as np
import torch
import soundfile
import random
import os
from scipy import signal
import glob

from src.datasets.base_dataset import BaseDataset


class TrainDataset(BaseDataset):
    def __init__(self, list_path, data_path, num_frames, musan_path, rir_path, start_label=0, name="train", *args, **kwargs):
        self.num_frames = num_frames
        self.start_label = start_label
		# Load and configure augmentation files
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-3] not in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

		# Load data & labels
        self.data_path = data_path
        self.list_path = list_path
        index = self._create_index_from_txt()
        super().__init__(index, *args, **kwargs)

    def _create_index_from_txt(self):
        index = []
        lines = open(self.list_path).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : (ii + self.start_label) for ii, key in enumerate(dictkeys) }
        for line in lines:
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(self.data_path, line.split()[1])
            index.append({
                    "data_path": file_name,
                    "label": speaker_label
                })
  
        return index

    
    def load_object(self, path):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(path)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = np.stack([audio],axis=0)
        # Data Augmentation
        augtype = random.randint(0,5)
        if augtype == 0:   # Original
            audio = audio
        elif augtype == 1: # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2: # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3: # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4: # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5: # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        data = torch.FloatTensor(audio[0])
        return data
        
    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float32), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = np.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_path = data_dict["data_path"]
        data_label = data_dict["label"]
        data_object = self.load_object(data_path)

        instance_data = {
            "data_object": data_object,
            "labels": data_label
        }

        return instance_data

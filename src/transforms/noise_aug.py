import torch
from torch import nn
import numpy as np
import soundfile
import random
from scipy import signal
import os
import glob


class NoiseAug(nn.Module):
    def __init__(self, num_frames, musan_path, rir_path):
        super().__init__()
		# Load and configure augmentation files
        self.num_frames = num_frames
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

    def forward(self, x):
        audio = x.numpy()
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
        x = torch.FloatTensor(audio[0])
        return x
    
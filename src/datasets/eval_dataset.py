import numpy as np
import torch
import soundfile

from src.datasets.base_dataset import BaseDataset


class EvalDataset(BaseDataset):
    def __init__(self, txt_path, data_path, name="test", *args, **kwargs):
        self.data_path = data_path
        self.txt_path = txt_path
        index = self._create_index_from_txt()
        super().__init__(index, *args, **kwargs)

    def _create_index_from_txt(self):
        index = []
        with open(self.txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                label, path1, path2 = parts
                index.append({
                    "label": int(label),
                    "path1": self.data_path + '/' + path1,
                    "path2": self.data_path + '/' + path2
                })
        return index

    
    def load_object(self, path):
        audio, _  = soundfile.read(path)

        data_1 = torch.FloatTensor(np.stack([audio], axis=0))

        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = np.stack(feats, axis = 0).astype(float)
        data_2 = torch.FloatTensor(feats)
        return (data_1, data_2)
        

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_path1 = data_dict["path1"]
        data_path2 = data_dict["path2"]
        data_object1_1, data_object1_2 = self.load_object(data_path1)
        data_object2_1, data_object2_2 = self.load_object(data_path2)
        data_label = data_dict["label"]

        instance_data = {"data_object1_1": data_object1_1, 
                         "data_object1_2": data_object1_2,
                         "data_object2_1": data_object2_1, 
                         "data_object2_2": data_object2_2,
                         "labels": data_label}

        return instance_data


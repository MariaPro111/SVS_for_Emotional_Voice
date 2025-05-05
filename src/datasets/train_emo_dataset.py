import numpy as np
import torch
import soundfile
import random
import os

from src.datasets.base_dataset import BaseDataset


class TrainEmoDataset(BaseDataset):
    def __init__(self, list_path, data_path, num_frames, start_label=0, name="train", *args, **kwargs):
        self.num_frames = num_frames
        self.start_label = start_label
		# Load data & labels
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(base_dir, "..", "..", data_path)
        self.list_path = os.path.join(base_dir, "..", "..", list_path)
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
            emo_label = int(line.split()[1])
            file_name = os.path.join(self.data_path, line.split()[2])
            index.append({
                    "path": file_name,
                    "label": speaker_label,
                    "emo_label": emo_label
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
        data = torch.FloatTensor(audio)
        return data
    
    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)
        data_label = data_dict["label"]
        data_emolabel = data_dict["emo_label"]

        instance_data = {"data_object": data_object, "labels": data_label, "emo_labels": data_emolabel}
        instance_data = self.preprocess_data(instance_data)

        return instance_data


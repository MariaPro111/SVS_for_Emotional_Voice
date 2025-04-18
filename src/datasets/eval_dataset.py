import numpy as np
import torch
import soundfile

from src.datasets.base_dataset import BaseDataset


class EvalDataset(BaseDataset):
    def __init__(self, data_list, data_path, name="test", *args, **kwargs):
        self.data_path = data_path
        self.data_list = data_list
        index = self._create_index_from_txt()
        super().__init__(index, *args, **kwargs)

    def _create_index_from_txt(self):
        path_idx = {}
        index = []
        files = []
        test_pairs = []
        lines = open(self.data_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))

        for i, path in enumerate(setfiles):
            path_idx[path] = i
        
        for line in lines:
            line = line.split()
            test_pairs.append([int(line[0]), path_idx[line[1]], path_idx[line[2]]])

        for path in setfiles:
            index.append({
                    "data_path": self.data_path + '/' + path,
                    "index": path_idx[path],
                    "test_pairs": test_pairs
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
        data_path = data_dict["data_path"]
        data_index = data_dict["index"]
        data_object_1, data_object_2 = self.load_object(data_path)
        test_pairs = data_dict["test_pairs"]

        instance_data = {"data_object_1": data_object_1, 
                         "data_object_2": data_object_2,
                         "index": data_index,
                         "test_pairs": test_pairs
                        } 
                         

        return instance_data


import requests
import zipfile
import os
import gdown


def download_voxceleb(url, target_dir, name):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, name)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(zip_path)

def download_lists():
    print("Downloading lists of pairs ...")

    gdown.download(id="1INOBiL7v4Vdrw1M8_4qGeOhOREt3Eoqg", output="data/voxceleb/Vox1_O.txt", quiet=False)   

    gdown.download(id="155VSN8b8R7Grm8_fp0hJ-PBBZbwynudJ", output="data/voxceleb/vox_train_list.txt", quiet=False)

    gdown.download(id="1JFJ0aT7OGgbjKVPkBCGi2lLO5G-2UJ8u", output="data/voxceleb/vox_train_emo_list.txt", quiet=False)




if __name__ == "__main__":
    print("Downloading test data VoxCeleb1 ...")
    os.makedirs("data/voxceleb/test", exist_ok=True)
    download_voxceleb('https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip', 'data/voxceleb/test', 'test.zip')
    print("Downloaded in data/voxceleb/test")

    print("Downloading train data VoxCeleb1 ...")
    os.makedirs("data/voxceleb/train", exist_ok=True)
    download_voxceleb('https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip', 'data/voxceleb/train', 'train.zip')
    print("Downloaded in data/voxceleb/train")

    download_lists()

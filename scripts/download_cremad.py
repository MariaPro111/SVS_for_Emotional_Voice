import os
import gdown
import zipfile

def download_and_extract_zip(url, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    zip_path = os.path.join(destination_folder, "data.zip")
    gdown.download(url, zip_path, quiet=False)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    os.remove(zip_path)


def download_data():
    file_url = "https://drive.google.com/uc?id=1meipD3ipWp_rJU9KsAiZofo5E5kaxCIK"
    os.makedirs("data/cremad", exist_ok=True)
    destination_folder = "data/cremad"    
    print(f"Downloading test data CREMA-D ...")
    download_and_extract_zip(file_url, destination_folder)
    print("Downloaded in data/cremad/crema-d_test")

    file_url = "https://drive.google.com/uc?id=1a-MU7FljuMy9ytX7-lyiSYhVrh-eRT-A"
    destination_folder = "data/cremad"    
    print(f"Downloading train data CREMA-D ...")
    download_and_extract_zip(file_url, destination_folder)
    print("Downloaded in data/cremad/crema_train")

def download_lists():
    print("Downloading lists of pairs ...")

    gdown.download(id="1HGomprLNgfZjYGJ5EB2SfWNFbvmF4Qz7", output="data/cremad/cremad_test_list.txt", quiet=False)

    gdown.download(id="1GeDzDu2i7WxbasSXjQqRTLKpThz7a8tt", output="data/cremad/train_list.txt", quiet=False)

    gdown.download(id="13Rg3cMmjOxEMqDhzatMZBafBkx4w95qC", output="data/cremad/train_emo_list.txt", quiet=False)


if __name__ == "__main__":
    download_data()
    download_lists()

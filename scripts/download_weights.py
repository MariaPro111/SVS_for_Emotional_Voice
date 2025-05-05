import os
import gdown


def download_weights():
    print("Downloading model weights ...")
    os.makedirs("data/weights", exist_ok=True)
    gdown.download(id="1osjV-smOK6Eezia5J6jJp4YE47vWpxE6", output="data/weights/wavlm_weights.pth", quiet=False)

    gdown.download(id="1WWo5faDvrjV-4BG1IqA8wRI4DBltJNNe", output="data/weights/wavlm_pretrained.pth", quiet=False)

    gdown.download(id="1uMlyRwa9h8OwK2ptAQdEouHAK8qNf3G1", output="data/weights/ecapatdnn_pretrained.model", quiet=False)

    gdown.download(id="1LDNBxp55sbyAtntUhzY8_j_jTPyLEsXI", output="data/weights/ecapatdnn_weights.pth", quiet=False)

if __name__ == "__main__":
    download_weights()
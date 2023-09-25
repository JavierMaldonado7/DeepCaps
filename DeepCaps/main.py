import torch
from torch.utils.data import DataLoader

from Decoder import CapsuleDecoder
from Encoder import Encoder
from Routing import Routing
from Train import train
import torch.optim as optim
import torch.nn as nn
from CapsuleNet import CapsuleNet
from Scan import Scan, device
from Datasets import OpenImagesDataset
from Datasets import Youtube8MDataset
from Datasets import DeepfakeDataset

# Define the device
device = device
# Define the network
capsule_net = CapsuleNet((3, 224, 224), num_classes=2)


def Detect(Routing, Path):
    routing = Routing

    test_video_path = Path

    scan = Scan(capsule_net.encoder, capsule_net.decoder, routing, test_video_path)

    result, confidence = scan.detect_deepfake()

    print(result, confidence)


def Train():
    # Define batch size for training(adjust for computing resources)
    batch_size = 200

    # Datasets defined
    train_dataOI = OpenImagesDataset
    train_dataYT = Youtube8MDataset
    test_data = DeepfakeDataset

    # train_dataset = Youtube8MDataset(data_dir="data", split="train")
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the amount of epochs the set is trained for
    epochs = 50

    # Defining save_path
    save_path = "Saved/"
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(capsule_net.parameters(), lr=0.001)
    print("Trained System!")
    # Train the network
# train(capsule_net, train_dataYT, test_data, device, epochs, optimizer, criterion, save_path)

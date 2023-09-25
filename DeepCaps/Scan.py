import cv2
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

# Verify which GPU is in place
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Scan:
    def __init__(self, encoder, decoder, routing, test_video_path):
        self.encoder = encoder
        self.decoder = decoder
        self.routing = routing
        self.test_video_path = test_video_path

    def detect_deepfake(self):
        # Load the test video and extract frames
        frames = self.extract_frames(self.test_video_path)

        # Define the transform to be applied on each frame
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Load the frames into a PyTorch DataLoader
        dataset = torch.utils.data.TensorDataset(frames)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

        # Set the model to evaluation mode
        self.encoder.eval()
        self.decoder.eval()

        # Initialize lists to store predictions and confidence scores
        predictions = []
        confidence_scores = []

        # Iterate over the frames in the test video and make predictions
        with torch.no_grad():
            for batch in dataloader:
                # Preprocess the batch of frames
                batch = batch[0].to(device)
                batch = transform(batch)

                # Encode the batch of frames
                encoded = self.encoder(batch)

                # Perform dynamic routing
                capsules = self.routing(encoded)

                # Decode the capsules
                decoded = self.decoder(capsules)

                # Calculate the reconstruction loss
                loss = torch.mean(torch.sum((decoded - batch) ** 2, dim=(1, 2, 3)))

                # Calculate the confidence scores for each sample in the batch
                confidence = torch.sqrt(torch.sum(capsules ** 2, dim=2))
                confidence_scores.extend(confidence.cpu().detach().numpy())

                # Classify the samples based on the confidence scores
                predictions.extend((confidence >= 0.5).cpu().detach().numpy())

        # Calculate the overall confidence score for the test video
        confidence_score = sum(confidence_scores) / len(confidence_scores)

        # Classify the test video as real or fake based on the predictions
        if all(predictions):
            return "Real", confidence_score
        else:
            return "Fake", confidence_score

    def extract_frames(self, video_path):
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Initialize list to store frames
        frames = []

        # Read frames from the video file
        while True:
            success, frame = video.read()
            if not success:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        # Release the video file
        video.release()

        return torch.stack([transforms.ToTensor()(frame) for frame in frames])

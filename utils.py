import torch
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def get_prediction(output):
    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)
    return pred.item(), confidence.item()
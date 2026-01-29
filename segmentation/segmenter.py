import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2

class Segmenter:
    def __init__(self):
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def segment(self, frame):
        input_tensor = self.transform(frame).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]

        mask = output.argmax(0).byte().cpu().numpy()
        return mask

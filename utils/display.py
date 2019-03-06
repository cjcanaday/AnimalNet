import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from PIL import Image


class Display:
    def __init__(self, model, predict, directory, num, IMG_SIZE=200):
        self.predict = predict
        self.dir = directory
        self.num = num
        self.model = model
        self.IMG_SIZE = IMG_SIZE

        self.trsfm = transforms.Compose([
            transforms.ToPILImage,
            transforms.Resize((self.IMG_SIZE,self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        images = DataLoader(self.dir, shuffle=True)
    def actual_display(self):
        i = 1
        while i < self.num:
            for img in images:
                path = os.path.join(self.dir,img)
                img = self.trsfm(Image.open(path))
                pred = self.model(img)
                print("For the image below the ConvNet guessed that it was a {}").format(pred)
                Image.open(path)
            


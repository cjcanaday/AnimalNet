import torch.nn as nn
from base import BaseModel
import torchvision.models as models
'''
Pad with: Input Size + (2 * Pad Size) - (Filter Size - 1)
'''

class CatDogModel(BaseModel):
    def __init__(self, num_classes=2):
        super(CatDogModel, self).__init__()
        self.layer1 = nn.Sequential(
            #One
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), # Batch x 16 x 50 x 50
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # Batch x 16 x 25 x 25 (50)
            #Two
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Batch x 32 x 50 x 50
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # Batch x 32 x 26 x 26
            nn.Dropout2d(0.25)

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2), # Batch x 64 x 26 x 26
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2), # Batch x 128 x 26 x 26
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # batch x 128 x 24 x 24 (14)
            nn.Dropout2d(0.25)


            

        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2), # Batch x 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, padding=2), # Batch x 512 x 14 x 14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # batch x 512 x 8 x 8
            nn.Dropout2d(0.25),
            
        )
        

        self.fc = nn.Sequential(
            nn.Linear(8*8*512,num_classes),
            nn.Dropout2d(0.50),
            nn.LogSoftmax()
            

        )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out






class PretrainedCatDog(BaseModel):
    def __init__(self):
        super(PretrainedCatDog, self).__init__()
    def make_model(self):
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        return model_ft



from torchvision import datasets, transforms
from base import BaseDataLoader


class CatDogDataLoader(BaseDataLoader):
    """
    Cat and Dog data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, color, training=True, IMG_SIZE=200):
        self.IMG_SIZE = IMG_SIZE
        self.color = color
        if self.color:
            image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            image = transforms.Grayscale(1)
        trsfm = transforms.Compose([
            transforms.Resize((self.IMG_SIZE,self.IMG_SIZE)),
            transforms.ToTensor(),
            image,

            ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=trsfm)
        super(CatDogDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
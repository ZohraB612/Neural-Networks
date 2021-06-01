import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToPILImage, Compose, ToTensor, Resize, Normalize, RandomHorizontalFlip, \
    RandomVerticalFlip, CenterCrop, ColorJitter

# sample is % of data to use
def get_loaders(sample=1, normalize=True):    
    DIR = "./dataset"
    DIR_TRAIN = DIR + "/training_set"
    DIR_TEST = DIR + "/test_set"
    
    # We check the format of the Data
    DIR_TRAIN_CAT = DIR_TRAIN + "/cats"
    DIR_TRAIN_DOG = DIR_TRAIN + "/dogs"
    
    DIR_TEST_CAT = DIR_TEST + "/cats"
    DIR_TEST_DOG = DIR_TEST + "/dogs"
    
    imgs_cat = os.listdir(DIR_TRAIN_CAT)
    imgs_dog = os.listdir(DIR_TRAIN_DOG)
    
    test_imgs_cat = os.listdir(DIR_TEST_CAT)
    test_imgs_dog = os.listdir(DIR_TEST_DOG)
    
    print('Cats:')
    print('Training set cat images format', imgs_cat[:5])
    print('Number of images in cat train', len(imgs_cat))
    print('Test set cat images format', test_imgs_cat[:5])
    print('==============================================')
    print('Dogs:')
    print('Training set dog images format', imgs_dog[:5])
    print('Number of images in dog train', len(imgs_dog))
    print('Test set dog images format', test_imgs_dog[:5])
    
    # We check the size of the full training folder (containing both classes)
    train_folder = ImageFolder(DIR_TRAIN)
    print("Original train ", len(train_folder))
    
    # data Augmentation
    trans = [Resize((128, 128)), ToTensor(), RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), CenterCrop(224), ColorJitter()]
    trans1 = [Resize((128, 128)), ToTensor()]
    if normalize:
        trans.append(Normalize(0.5, 0.5, 0.5))
        trans1.append(Normalize(0.5, 0.5, 0.5))
    train_folder = ImageFolder(DIR_TRAIN, transform=Compose(trans))
    
    sample_ids = list(range(0, len(train_folder), int(1.0 / sample)))
    sampled_train = Subset(train_folder, sample_ids)

    test_folder = ImageFolder(DIR_TEST, transform=Compose(trans1))
    test_loader = DataLoader(test_folder, batch_size=128)

    train_idx, valid_idx = train_test_split(list(range(len(sampled_train))), test_size=0.2)
    
    train_dataset = Subset(sampled_train, train_idx)
    valid_dataset = Subset(sampled_train, valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=128)
    valid_loader = DataLoader(valid_dataset, batch_size=128)

    return train_loader, valid_loader, test_loader

"""
used to plot sample images
"""

def plot_sample(train_loader):

    inputs, classes = next(iter(train_loader))
    catId = list(classes).index(0)
    dogId = list(classes).index(1)
    cat = ToPILImage()(inputs[catId]).convert("RGB")
    dog = ToPILImage()(inputs[dogId]).convert("RGB")

    plt.imshow(cat)
    plt.show()
    plt.imshow(dog)
    plt.show()

if __name__ == "__main__":
    train_loader, _, _ = get_loaders(sample=0.01, normalize=False)
    plot_sample(train_loader)

import os
import cv2
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from preprocessing import augmentations
from sklearn.model_selection import train_test_split



# Класс создания датасета для обучения и валидации
class DogsVSCatsDataset(Dataset):
    def __init__(self, images_filepaths, test_flag = False, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform
        self.test_flag = test_flag

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        if self.test_flag:           
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image
        else:
            if os.path.normpath(image_filepath).split(os.sep)[-2] == "cats":
                label = 1.0
            else:
                label = 0.0
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image, label

# Функция для создания датасета
def create_loader(correct_images_filepaths):
    random.shuffle(correct_images_filepaths)
    train_images_filepaths, val_images_filepaths = train_test_split(correct_images_filepaths, train_size=0.6)
    val_images_filepaths, test_images_filepaths = train_test_split(val_images_filepaths,test_size=0.1)

    train_dataset = DogsVSCatsDataset(images_filepaths=train_images_filepaths, test_flag=False ,transform=augmentations('base'))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = DogsVSCatsDataset(images_filepaths=val_images_filepaths, transform=augmentations('base'))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = DogsVSCatsDataset(images_filepaths=test_images_filepaths, test_flag=True, transform=augmentations('test'))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


    return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset, test_images_filepaths

import os
import cv2
from torch.utils.data import Dataset


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



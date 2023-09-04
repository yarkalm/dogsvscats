import copy
import random
import albumentations as A
from matplotlib import pyplot as plt
from dataset import DogsVSCatsDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


def visualize_augmentations(dataset, idx=0, samples=8, cols=4):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        print(dataset[idx])
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def create_loader(correct_images_filepaths):
    random.shuffle(correct_images_filepaths)

    train_images_filepaths, val_images_filepaths = train_test_split(correct_images_filepaths, train_size=0.6)
    val_images_filepaths, test_images_filepaths = train_test_split(val_images_filepaths,test_size=0.1)

    train_dataset = DogsVSCatsDataset(images_filepaths=train_images_filepaths, test_flag=False ,transform=base_augmentations)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = DogsVSCatsDataset(images_filepaths=val_images_filepaths, transform=base_augmentations)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    #visualize_augmentations(train_dataset)
    test_dataset = DogsVSCatsDataset(images_filepaths=test_images_filepaths, test_flag=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


    return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset


base_augmentations = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.CenterCrop(height=128, width=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
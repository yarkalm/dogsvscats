import os
import cv2
import torch
from pathlib import Path
from preprocessing import create_loader
from evaluate import predict, display_image_grid
from classifier import create_model, train, validate

if __name__ == '__main__':
    # Создадим основные переменные основных директорий
    dataset_directory = Path('dataset')
    cats_directory = dataset_directory / 'train/cats'
    dogs_directory = dataset_directory / 'train/dogs'

    # Список файлов для дальнейшей предообработки
    cats_images_filepaths = sorted([cats_directory / f for f in os.listdir(cats_directory)])
    dogs_images_filepaths = sorted([dogs_directory / f for f in os.listdir(dogs_directory)])
    images_filepaths = [*cats_images_filepaths, *dogs_images_filepaths]
    correct_images_filepaths = [str(i) for i in images_filepaths if cv2.imread(str(i)) is not None]
    test_images_filepaths = correct_images_filepaths[-10:]

    # Список параметров для НС
    params = {
        "model": "resnet50",
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "lr": 0.001,
        "epochs": 10,
    }

    # Создание экземпляра модели
    model, criterion, optimizer = create_model(params)
    # Создание датасетов
    train_loader, val_loader, test_loader = create_loader(correct_images_filepaths)

    # Запуск обучения
    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        validate(val_loader, model, criterion, epoch, params)

    # Тестирование
    predicted_labels = predict(model, params, test_loader)
    display_image_grid(test_images_filepaths, predicted_labels)

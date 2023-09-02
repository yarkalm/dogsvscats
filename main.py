import os
import cv2
import torch
from pathlib import Path
from preprocessing import create_loader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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
    # correct_images_filepaths = [str(i) for i in images_filepaths if cv2.imread(str(i)) is not None]
    correct_images_filepaths = []
    for i in images_filepaths:
        if cv2.imread(str(i)) is not None:
            print(i)
            correct_images_filepaths.append(str(i))
    test_images_filepaths = correct_images_filepaths[-10:]

    # Список параметров для НС
    params = {
        "model": "resnet50",
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "lr": 0.001,
        "epochs": 2,
    }

    # Создание экземпляра модели
    model, criterion, optimizer = create_model(params)
    # Создание датасетов
    train_loader, val_loader, test_loader = create_loader(correct_images_filepaths)

    # Запуск обучения
    for epoch in range(1, params["epochs"] + 1):
        history_train = train(train_loader, model, criterion, optimizer, epoch, params)
        history_test = validate(val_loader, model, criterion, epoch, params)

    # Тестирование
    predicted_labels = predict(model, params, test_loader)
    print((test_images_filepaths),(predicted_labels))
    display_image_grid(test_images_filepaths, predicted_labels)

    y_true_test = []
    for batch in test_loader:
        inputs, labels = batch
        y_true_test.append(torch.tensor(labels))
    print(y_true_test)
    print(predicted_labels)
    conf_matrix = confusion_matrix(y_true_test, predicted_labels)
    print("Confusion Matrix of the Test Set")
    print("-----------")
    print(conf_matrix)
    print("Precision of the Model :\t" + str(precision_score(y_true_test, predicted_labels)))
    print("Recall of the Model    :\t" + str(recall_score(y_true_test, predicted_labels)))
    print("F1 Score of the Model  :\t" + str(f1_score(y_true_test, predicted_labels)))

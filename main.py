import os
import cv2
import torch
import itertools
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from preprocessing import create_loader
from evaluate import predict, display_image_grid, conf_matrix
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
    train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset, y_true_dataset, y_true_loader = create_loader(correct_images_filepaths)

    # Запуск обучения
    history_training = {'accuracy':[],'loss':[]}
    history_validation = {'accuracy':[],'loss':[]}
    for epoch in range(1, params["epochs"] + 1):
        history_train = train(train_loader, model, criterion, optimizer, epoch, params)
        history_training['accuracy'].append((history_train[0]))
        history_training['loss'].append((history_train[1]))

        history_val = (validate(val_loader, model, criterion, epoch, params))
        history_validation['accuracy'].append((history_val[0]))
        history_validation['loss'].append((history_val[1]))


    # Вывод графиков
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(121)

    ax.plot(list(map(np.mean, history_training['accuracy'])), label='train')
    ax.plot(list(map(np.mean, history_validation['accuracy'])), label='test')
    ax.set_title('График точности "Accuracy"')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    ax.grid(True)


    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(121)
    ax.plot(list(map(np.mean, history_training['loss'])), label='train')
    ax.plot(list(map(np.mean, history_validation['loss'])), label='test')
    ax.set_title('График функции потерь "Loss"')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    ax.grid(True)
    plt.show()



    
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        (This function is copied from the scikit docs.)
        """
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", 
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    # Тестирование
    predicted_labels = predict(model, params, test_loader)
    print((test_images_filepaths),(predicted_labels))
    display_image_grid(test_images_filepaths, predicted_labels)

    conf_matrix(model, params, train_loader)

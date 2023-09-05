import os
import cv2
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



# Функция расчёта точности
def cal_acc(output, target, sigma):
    output = torch.sigmoid(output) >= sigma
    target = target == 1.0
    return multiclass_f1_score(torch.squeeze(output, dim=1), torch.squeeze(target, dim=1), num_classes=2).item()



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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

# вывод изображений
def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2][:3]
        predicted_label = predicted_labels[i] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

# функция предсказания
def predict(model, params, test_loader):
    model = model.eval()
    predicted_labels = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            predictions = (torch.sigmoid(output) >= 0.5)[:, 0].cpu().numpy()
            predicted_labels += ["cat" if is_cat else "dog" for is_cat in predictions]

    return predicted_labels

def conf_matrix(model, params, loader):
    model = model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(params["device"], non_blocking=True)
            labels = labels.to(params["device"], non_blocking=True)
            output = model(images)

            y_pred.extend((torch.sigmoid(output) >= 0.5)[:, 0].cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    cm = (confusion_matrix(y_true, y_pred))
    plot_confusion_matrix(cm,['cats','dogs'])

    print("Precision of the Model :\t" + str(precision_score(y_true, y_pred)))
    print("Recall of the Model    :\t" + str(recall_score(y_true, y_pred)))
    print("F1 Score of the Model  :\t" + str(f1_score(y_true, y_pred)))

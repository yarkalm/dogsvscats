import os
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score




# вывод изображений
def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
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
            predicted_labels += ["Cat" if is_cat else "Dog" for is_cat in predictions]

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

            y_pred.extend(torch.sigmoid(output) >= 0.5)[:, 0].cpu().numpy()
            y_true.extend(labels.cpu().numpy())
    
    print(confusion_matrix(y_true, y_pred))
    print("Precision of the Model :\t" + str(precision_score(y_true, y_pred)))
    print("Recall of the Model    :\t" + str(recall_score(y_true, y_pred)))
    print("F1 Score of the Model  :\t" + str(f1_score(y_true, y_pred)))


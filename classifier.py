import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models
from evaluate import cal_acc
from metrics import MetricMonitor


# Функция создания модели
def create_model(params, ):
    model = getattr(models, params["model"])(pretrained=False, num_classes=1, )
    model = model.to(params["device"])
    criterion = nn.BCEWithLogitsLoss().to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    return model, criterion, optimizer


# Функция запуска обучения
def train(train_loader, model, criterion, optimizer, epoch, params):
    acc_history = []
    loss_history = []
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True).float().view(-1, 1)
        output = model(images)
        loss = criterion(output, target)
        accuracy = cal_acc(output, target, params['sigma'])
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc_history.append(accuracy)
        loss_history.append(loss.item())
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
    return [acc_history, loss_history]


# Функция запуска валидации
def validate(val_loader, model, criterion, epoch, params):
    acc_history = []
    loss_history = []
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            accuracy = cal_acc(output, target, params['sigma'])
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            acc_history.append(accuracy)
            loss_history.append(loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
    return [acc_history, loss_history]
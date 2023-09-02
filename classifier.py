import torch
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict
import torchvision.models as models
from torcheval.metrics.functional import multiclass_f1_score




class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

# Функция расчёта точности
def cal_acc(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return multiclass_f1_score(torch.squeeze(output, dim=1), torch.squeeze(target, dim=1), num_classes=2).item()

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
        accuracy = cal_acc(output, target)
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
            accuracy = cal_acc(output, target)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            acc_history.append(accuracy)
            loss_history.append(loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
    return [acc_history, loss_history]

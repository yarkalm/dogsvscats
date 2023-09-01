import torch
import torch.nn as nn
import torch.optim as optim
from classifier import DogCatClassifier
from preprocessing import train_loader, val_loader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DogCatClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass and compute loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the evaluation set: {100 * correct / total:.2f}%')



train_model(model, criterion, optimizer, num_epochs=8)
evaluate_model(model, val_loader)

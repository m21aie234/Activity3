#https://colab.research.google.com/drive/1MI6Buny5t1wQztSHnbxSlNyi3Kn0UFUR?usp=sharing
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import STL10, SVHN, FashionMNIST
from torchvision.utils import make_grid
from tqdm import tqdm

# Check the last digit of your roll number for selecting the dataset
# roll_number = 234  # Replace with your roll number
dataset_options = ['STL10', 'SVHN', 'FashionMNIST']
#selected_dataset = dataset_options[roll_number % 3]

# Define transforms based on the selected dataset
#if selected_dataset == 'STL10':
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# num_classes = 10
# train_dataset = STL10(root='./data', split='train', download=True, transform=train_transform)
# test_dataset = STL10(root='./data', split='test', download=True, transform=test_transform)

# elif selected_dataset == 'SVHN':
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
num_classes = 10
train_dataset = SVHN(root='./data', split='train', download=True, transform=train_transform)
test_dataset = SVHN(root='./data', split='test', download=True, transform=test_transform)

# else:  # FashionMNIST
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     num_classes = 10
#     train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
#     test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet101 model
model = models.resnet101(pretrained=True)

# Modify the last fully connected layer to adapt to the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Define optimizers
optimizers = {
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'Adagrad': optim.Adagrad(model.parameters(), lr=0.001),
    'Adadelta': optim.Adadelta(model.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001)
}

# Define loss function
criterion = nn.CrossEntropyLoss()

# Training function
def train_model(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Initialize lists to store results
train_losses = {optimizer_name: [] for optimizer_name in optimizers}
train_accuracies = {optimizer_name: [] for optimizer_name in optimizers}

# Train the model with each optimizer
num_epochs = 5  # Adjust the number of epochs as needed
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    for optimizer_name, optimizer in optimizers.items():
        train_loss, train_acc = train_model(model, criterion, optimizer, train_loader)
        train_losses[optimizer_name].append(train_loss)
        train_accuracies[optimizer_name].append(train_acc)
        print(f'{optimizer_name} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')

# Plot training curves
plt.figure(figsize=(12, 5))
for optimizer_name in optimizers:
    plt.plot(train_losses[optimizer_name], label=f'{optimizer_name} Loss')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
for optimizer_name in optimizers:
    plt.plot(train_accuracies[optimizer_name], label=f'{optimizer_name} Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy (%)')
plt.title('Training Accuracy vs Epoch')
plt.legend()
plt.show()

# Evaluate the model on the test set
model.eval()
top5_accuracy = 0.0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = outputs.topk(5, 1)  # Get top-5 predictions
        labels = labels.view(-1, 1).expand_as(predicted)
        top5_accuracy += (predicted == labels).sum().item()
        total += labels.size(0)

# Calculate top-5 accuracy
final_top5_accuracy = 100.0 * top5_accuracy / total
print(f'Final Top-5 Test Accuracy: {final_top5_accuracy:.2f}%')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import *
import matplotlib.pyplot as plt

train_data, test_data = main_cnn('task1/seg_train/seg_train', 'task1/seg_test/seg_test')

# Set up dataloaders for training and test data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 1 input channel for grayscale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 25 * 25, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
# Create the CNN
model = CNN(num_classes=6)
# Define criterion for error calculation
criterion = nn.CrossEntropyLoss()
# Choose optimizing method
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_CNN(model, train_loader, criterion, optimizer, num_epochs):
    # Moving model training to GPU to speed up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to GPU

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation loop with tqdm
def evaluate_CNN(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Evaluating", unit="batch") as pbar:
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

train_accuracies = []
test_accuracies = []

def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to GPU

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0
        correct, total = 0, 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        correct = 0
        correct, total = 0, 0
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc="Evaluating", unit="batch") as pbar:
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                    outputs = model(inputs)

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar.update(1)

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Test Accuracy: {accuracy:.2f}%")

# Train and evaluate the model
'''train_CNN(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_CNN(model, test_loader)'''

train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, 20)

epochs = range(1, len(train_accuracies) + 1)  # Create a range starting from 1

plt.plot(epochs, train_accuracies, label="Training accuracy", color="blue")
plt.plot(epochs, test_accuracies, label="Test accuracy", color="orange")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Epochs')
plt.show()

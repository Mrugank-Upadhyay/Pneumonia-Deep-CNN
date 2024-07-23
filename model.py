import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from PIL import Image
import pandas as pd
from tqdm import tqdm
import seaborn as sbn

# Data directory
DATA_DIR = os.getcwd() + "/archive/"
SAVE_PATH = "./cnn_net.pth"

# Hyperparams
DATALOADER_BATCH_SIZE = 32
LEARNING_RATE = 0.002
EPOCH = 26

class Image_EDA:
    def __init__(self, path):
        self.directories = {0: DATA_DIR + "{}bacteria/".format(path),
                            1: DATA_DIR + "{}normal/".format(path),
                            2: DATA_DIR + "{}virus/".format(path)}
        self.classes_len = {0: len(os.listdir(self.directories[0])),
                            1: len(os.listdir(self.directories[1])),
                            2: len(os.listdir(self.directories[2]))}

    def class_directories(self):
        return self.directories

    def class_dir_len(self):
        return self.classes_len

    def class_balance(self):
        fig = plt.figure("Train Data Balance")
        plt.bar(self.classes_len.keys(),
                self.classes_len.values(), width=0.5)
        plt.title("Train Data Balance")
        plt.xlabel("Classes")
        plt.ylabel("# of images")
        plt.show()

    def display_image_sizes(self):
        fig = plt.figure()
        
        # # Show plots of size for each class
        for n, d in tqdm(self.directories.items()):
            filepath = d
            filelist = [filepath + f for f in os.listdir(filepath)]
            colors = []
            height = []
            width = []
            for i in tqdm(range(image_eda.class_dir_len()[n])):
                image = Image.open(filelist[i])
                arr = np.array(image)
                size = arr.shape
                colors.append('red' if n == 0 else 'green' if n == 1 else 'blue')
                height.append(size[0])
                width.append(size[1])
            plt.scatter(x=width, y=height, c=colors)
        plt.title("Image size of classes")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.legend(["Bacteria", "Normal", "Virus"])
        plt.show()              



image_eda = Image_EDA("train/")
# image_eda.class_balance()
# image_eda.display_image_sizes()


class Load_Data:
    def load(self, path):
        transform = transforms.Compose([transforms.Resize(250),
                                        transforms.CenterCrop(200),
                                        transforms.Grayscale(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(90),
                                        transforms.ToTensor()
                                        ])
        images = datasets.ImageFolder(
            root=DATA_DIR + path, transform=transform)
        dataloader = torch.utils.data.DataLoader(
            images, batch_size=DATALOADER_BATCH_SIZE, shuffle=True)
        return dataloader


load_data = Load_Data()

train_dataloader = load_data.load("train/")

test_dataloader = load_data.load("test/")

val_dataloader = load_data.load("val/")

train_loader_len = len(train_dataloader.dataset)
test_loader_len = len(test_dataloader.dataset)
val_loader_len = len(val_dataloader.dataset)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 16, kernel_size=2, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.LazyLinear(50),
            nn.ReLU(inplace=True),
            nn.LazyLinear(3),
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x
    
    def train(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.train_acc = []
        self.train_loss = []
        loss_sample_freq = np.floor((train_loader_len / DATALOADER_BATCH_SIZE) / 5)
        for epoch in tqdm(range(EPOCH)):
            running_loss = 0.0
            total = 0.0
            correct = 0.0
            for i, data in tqdm(enumerate(train_dataloader, 0)):
                images, labels = data
                
                # zero the optimer gradients
                self.optimizer.zero_grad()

                output = self(images)
                loss = self.criterion(output, labels)
                # zero the network gradients
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if i % loss_sample_freq == 0 and i != 0:
                    acc = correct / total * 100
                    self.train_acc.append(acc)
                    self.train_loss.append(running_loss / loss_sample_freq)
                    print(f"Epoch: {epoch + 1}, iteration = {i+1}, loss = {running_loss / loss_sample_freq:.5f}")
                    running_loss = 0.0 if i <= loss_sample_freq * 5 else running_loss
        acc = correct / total * 100
        self.train_acc.append(acc)
        print(f'Accuracy of the network on the training images: {acc} %')

        print("Done Training")
        torch.save(self.state_dict(), SAVE_PATH)
    
    def test(self):
        self.eval_acc = []
        self.eval_loss = []
        total_output = []
        for epoch in tqdm(range(EPOCH)):
            # since we're not training, we don't need to calculate the gradients for our outputs
            correct = 0
            total = 0
            running_loss = 0.0
            loss_sample_freq = np.floor((val_loader_len / DATALOADER_BATCH_SIZE) / 5)
            with torch.no_grad():
                for i, data in tqdm(enumerate (val_dataloader, 0)):
                    images, labels = data
                    # calculate outputs by running images through the network
                    output = self(images)
                    total_output.extend(output)
                    loss = self.criterion(output, labels)
                    running_loss += loss.item()

                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if i % loss_sample_freq == 0 and i != 0:
                        acc = correct / total * 100
                        self.eval_acc.append(acc)
                        self.eval_loss.append(running_loss / loss_sample_freq)
                        print(f"Epoch: {epoch + 1}, iteration = {i+1}, loss = {running_loss / loss_sample_freq:.5f}")
                        running_loss = 0.0 if i <= loss_sample_freq * 5 else running_loss
            acc = correct / total * 100
            # test_loss = running_loss / test_loader_len
            self.eval_acc.append(acc)
            # self.eval_loss.append(test_loss)

        acc = correct / total * 100
        print(f'Accuracy of the network on the test images: {acc} %')
        return total_output
    
    def plot_loss(self):
        print(f"train loss = {self.train_loss}")
        print(f"eval loss = {self.eval_loss}")
        plt.figure("Loss")
        plt.plot(self.train_loss,'-o')
        plt.plot(self.eval_loss,'-o')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Loss')
        plt.show()
        

    def plot_acc(self):
        print(f"train acc = {self.train_acc}")
        print(f"eval acc = {self.eval_acc}")
        plt.figure("Accuracy")
        plt.plot(self.train_acc,'-o')
        plt.plot(self.eval_acc,'-o')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Accuracy')
        plt.show()
        

net = Net()
summary(net, (1,200,200))
net.train()
predictions = net.test()
net.plot_loss()
net.plot_acc()

ground_truth = []

for data in val_dataloader:
    images, labels = data
    ground_truth.extend(labels)

class_names = ['bacteria', 'normal', 'virus']
cm = confusion_matrix(ground_truth, predictions)
plt.figure(figsize=(8,8))
plt.title("Confusion Matrix")
sbn.heatmap(cm, cbar=False, xticklabels=class_names, yticklabels=class_names, fmt='d', annot=True, cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show
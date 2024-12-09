'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------

NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email:
yliu9097@uni.sydney.edu.au, yili7216@uni.sydney.edu.au
'''

# Import necessary packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim

# Import the custom network architecture
from network import Network 

# Set up argument parser for command-line options
parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()


# Define an EarlyStopper class for early stopping during training
# from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    """
    Implements early stopping to terminate training when validation loss stops improving.
    """

    def __init__(self, patience=1, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')


    def early_stop(self, validation_loss):
        """
        Checks if training should be stopped early.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_net(net, trainloader, valloader):
    """
    Trains the neural network and evaluates it on the validation set.

    Args:
        net (nn.Module): The neural network model to train.
        trainloader (DataLoader): DataLoader for the training data.
        valloader (DataLoader): DataLoader for the validation data.

    Returns:
        int: The best validation accuracy achieved during training.
    """

    epochs = 20  # Number of training epochs
    lr = 0.001  # Learning rate
    save_path = 'project2.pth'

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.AdamW(net.model.fc.parameters(), lr=lr, weight_decay=0.0001, eps=1e-7, amsgrad=True)
    optimizer = optim.AdamW(net.model.classifier.parameters(), lr=lr, weight_decay=0.0001, eps=1e-7, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_acc = 0.0
    early_stopper = EarlyStopper(patience=3, min_delta=0.25)     # Initialize early stopper

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in trainloader:
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += torch.sum(preds == labels)
            total_train += labels.size(0)

        train_acc = (correct_train.float() / total_train) * 100 # Calculate training accuracy

        # Validation phase
        val_loss, val_acc = validate_net(net, valloader)

        # Check for early stopping
        if early_stopper.early_stop(val_loss):
            break

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with validation accuracy: {val_acc:.2f}%")

        scheduler.step()  # Update the learning rate

    val_accuracy = int(float(f"{best_val_acc:.2f}"))
    return val_accuracy


def validate_net(net, valloader):
    """
    Evaluates the neural network on the validation set.

    Args:
        net (nn.Module): The neural network model to evaluate.
        valloader (DataLoader): DataLoader for the validation data.

    Returns:
        tuple: A tuple containing the validation loss and accuracy.
    """

    net.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in valloader:
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_val += torch.sum(preds == labels)
            total_val += labels.size(0)

    val_acc = (correct_val.float() / total_val) * 100
    print(f"Validation Loss: {val_loss/len(valloader):.4f}, Validation Accuracy: {val_acc:.2f}%")
    return val_loss, val_acc


##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

####################################

####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

train_image_path = '../train/'
validation_image_path = '../validation/'

trainset = ImageFolder(train_image_path, train_transform)
valset = ImageFolder(validation_image_path, test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                         shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'.

# Initialize the network
network = Network()
if args.cuda:
    network = network.cuda()

# Train the network and evaluate on the validation set
val_acc = train_net(network, trainloader, valloader)

print("final validation accuracy:", val_acc)

# ==================================

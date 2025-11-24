import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import os

import RNNsetup as rs
from spamclassifier import evaluate

BATCH_SIZE = 64

#Create the datasets (see RNNsetup.py)
trainDataset, testDataset, vocab = rs.createDatasets()

#Create dataloaders
trainLoader = DataLoader(trainDataset, batch_size = BATCH_SIZE, shuffle = True)
testLoader = DataLoader(testDataset, batch_size = BATCH_SIZE, shuffle = False)

#Create the model
vocabSize = len(vocab)
outputSize = 2

model = None
path = None

#Model selection
while model == None:
    modelType = input("Enter model type (RNN or LSTM): ")
    if modelType.lower() == "rnn":
        embedSize = 128
        hiddenSize = 256
        model = rs.SimpleRNN(vocabSize, embedSize, hiddenSize, outputSize)
        path = "../simpleRNN_trained.pth"
    elif modelType.lower() == "lstm":
        embedSize = 512
        hiddenSize = 512
        model = rs.LSTM(vocabSize, embedSize, hiddenSize, outputSize)
        path = "../LSTM_trained.pth"
    else:
        print("Incorrect model type entered. Please enter RNN or LSTM.")

#Define the loss and optimization functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Send model to the GPU, if one is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Training loop
if not os.path.exists(path):
    print("Training...")
    numEpochs = 20
    for epoch in range(numEpochs):
        #Testing accuracy each epoch
        correct = 0
        total = 0
        
        model.train()
        epochLoss = 0
        #Loop through the data batch by batch (BATCH_SIZE defined above)
        for emails, labels in trainLoader:
            #Send data to GPU
            emails = emails.to(device)
            labels = labels.to(device)

            #Forward pass
            outputs = model(emails)
            loss = criterion(outputs, labels)
            
            #Back propagation and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epochLoss += loss.item()

            #For accuracy calc
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{numEpochs}], Loss: {epochLoss / len(trainLoader):.4f}, Accuracy: {100 * correct / total:.2f}')

    #Save the trained model
    torch.save(model.state_dict(), path)
    print("Model saved")
else:
    #Load the model if it's already been trained
    model.load_state_dict(torch.load(path))
    print("Model loaded")


#Model evaluation (using test set)
print("\nEvaluating...")
model.eval()
correct = 0
total = 0

allLabels = []
allPreds = []
allProbs = []

with torch.no_grad():
    for emails, labels in testLoader:
        emails = emails.to(device)
        labels = labels.to(device)

        outputs = model(emails)

        #For accuracy calc
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        #Collect information for the metrics
        probs = torch.softmax(outputs, dim=1)[:, 1]
        allLabels.extend(labels.cpu().numpy())
        allPreds.extend(predicted.cpu().numpy())
        allProbs.extend(probs.cpu().numpy())

allLabels = np.array(allLabels)
allPreds = np.array(allPreds)
allProbs = np.array(allProbs)

evaluate(allLabels, allPreds, allProbs, "RNN")
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from collections import Counter

from spamclassifier import importDataset

MIN_FREQUENCY = 3

#Dataset class for PyTorch
class EmailDataset(Dataset):
    #Initialize the dataset w/ emails and labels
    def __init__(self, encodedEmails, labels):
        self.emails = encodedEmails
        self.labels = labels

    #Returns the length of the dataset
    def __len__(self):
        return len(self.emails)
    
    #Gets tensors of a specific (email, label) tuple
    def __getitem__(self, index):
        email = torch.tensor(self.emails[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return email, label

#Class for the simple many to one RNN and the foward pass. Code replicated from https://www.geeksforgeeks.org/deep-learning/implementing-recurrent-neural-networks-in-pytorch
class SimpleRNN(nn.Module):
    def __init__(self, vocabSize, embedSize, hiddenSize, outputSize):
        super(SimpleRNN, self).__init__()
        self.hiddenSize = hiddenSize
        #An embedding layer converts each word index into a dense vector,
        #which is learned over time so that similar vectors are usually words used in the same contexts
        self.embedding = nn.Embedding(vocabSize, embedSize)
        #The hidden RNN layer, which takes in a single embedded vector (1 word) and outputs a hidden state
        self.rnn = nn.RNN(embedSize, hiddenSize, batch_first=True)
        #Fully connected layer, which maps the final hidden state to the likelihood (as arbitrary logits)
        #that the email belongs to either class (output size = 2 for 2 classes)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        #Pass inputs through embedding layer
        x = self.embedding(x)
        #Create an initial hidden state of all zeros
        h0 = torch.zeros(1, x.size(0), self.hiddenSize).to(x.device)
        #Pass the batch through the RNN layer
        out, _ = self.rnn(x, h0)
        #Complete the fully connected pass using the final hidden state from the RNN pass
        out = self.fc(out[:, -1, :])
        return out

#Class for the LSTM. Code is similar to the simple RNN but replaces the RNN layer with a LSTM layer
class LSTM(nn.Module):
    def __init__(self, vocabSize, embedSize, hiddenSize, outputSize):
        super(LSTM, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(vocabSize, embedSize)
        #Add extra layers and a dropout to further improve performance
        self.lstm = nn.LSTM(embedSize, hiddenSize, batch_first=True, num_layers=3, dropout=0.5)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


#Create the vocab (word to integer mappings) for PyTorch
def createVocab(emails, minFrequency):
    #Count the frequency of each word in the data
    counter = Counter()
    for email in emails:
        counter.update(email.split())

    #Set up the vocab with mappings for padding and unknown words
    vocab = {"<PAD>": 0, "<UNK>": 1}

    #Add words above the frequency threshold to the vocab
    for word, frequency in counter.items():
        if frequency >= minFrequency:
            vocab[word] = len(vocab)

    return vocab

#Translate words in the emails to their integer values and pad (or cut) so they're all the same length
def encodeEmails(emails, vocab, maxLen):
    encodedEmails = []

    for email in emails:
        #Translate words to their integers
        encoded = [vocab.get(word, vocab["<UNK>"]) for word in email.split()]
        
        #Cut the end of the email if too big
        if len(encoded) >= maxLen:
            encoded = encoded[:maxLen]
        #Pad if too small
        else:
            encoded = encoded + [vocab["<PAD>"]] * (maxLen - len(encoded))

        encodedEmails.append(encoded)

    return encodedEmails

#Abstract away the dataset creation process
def createDatasets():
    #Use code from milestone 2 to import and process the dataset
    df = importDataset()

    #Set the max length for padding and cutting to the 99th percentile so 
    #there isn't too much padding or too many emails being cut
    lengths = [len(email.split()) for email in df["clean text"]]
    maxLen = int(np.percentile(lengths, 92))

    #Use previous functions to get encoded emails
    vocab = createVocab(df["clean text"], MIN_FREQUENCY)
    encodedEmails = encodeEmails(df["clean text"], vocab, maxLen)

    #Transform labels into integers (ham = 0, spam = 1)
    labelEncoder = LabelEncoder()
    intLabels = labelEncoder.fit_transform(df["label"])

    #Split dataset into train/test sets (80%/20%)
    seed = 4
    trainEmails, testEmails, trainLabels, testLabels = train_test_split(encodedEmails, intLabels, test_size=0.2, random_state=seed, stratify=intLabels)

    #Use dataset class to create the datasets
    trainDataset = EmailDataset(trainEmails, trainLabels)
    testDataset = EmailDataset(testEmails, testLabels)

    return trainDataset, testDataset, vocab
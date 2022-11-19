#Importing libraries and modules -
import json
import numpy as np
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, lemm, bag_of_words

#Load Intents module -
with open('intents.json','r') as f:
    intents = json.load(f)

#Converting intents.json files - entences to words -
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

#Ignoring the words -
ignore_words = ['?','!','.',',']
all_words = [lemm(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#Defining the training data for X and Y -
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)

#Converting set data to array form-
X_train = np.array(X_train)
y_train = np.array(y_train)

# Iteration and batch Trining -
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    #dataset[index] -
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
#Hyperparameters -
batch_size = 8
hidden_size = 8
learning_rate = 0.001
output_size = len(tags)
input_size = len(X_train[0])
num_epochs = 1500
    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#Condition if GPU available run on GPU and if not then run on CPU -
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Create model -
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#Compile the model -
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        #Forward Pass -
        outputs = model(words)
        loss = criterion(outputs, labels)

        #Backword Pass -
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

#Saving the Model and data in pickle file format -
data = {"model_state":model.state_dict(),"input_size":input_size,"output_size":output_size,
        "hidden_size":hidden_size,"all_words":all_words,"tags":tags}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training Completed, file saved to {FILE}')
import random
import numpy as np
import torch
import torch.nn as nn
import nltk
import json
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open('intents.json','r') as i:
  intents=json.load(i)

class NeuralNet(nn.Module):
  def __init__(self,input_size,hidden_size,num_classes):
    super(NeuralNet,self).__init__()
    self.l1=nn.Linear(input_size,hidden_size)
    self.l2=nn.Linear(hidden_size,hidden_size)
    self.l3=nn.Linear(hidden_size,num_classes)
    self.relu=nn.ReLU()
  def forward(self,x):
    out=self.l1(x)
    out=self.relu(out)
    out=self.l2(out)
    out=self.relu(out)
    out=self.l3(out)
    return out



def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
   tokenized_sentence=[stem(w) for w in tokenized_sentence]
   bag=np.zeros(len(all_words), dtype=np.float32)
   for idx, w in enumerate(all_words):
    if w in tokenized_sentence:
      bag[idx]=1.0
   return bag

FILE = "data.pth"
data = torch.load(FILE)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import random
from tqdm.notebook import tqdm
import timeit
from timeit import default_timer as timer

from helper_functions import accuracy_fn
import TinyVGG

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

# Dataloaders
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# Model
FashionMNIST_Model = TinyVGG(input_shape=1,
                             hidden_units=50,
                             output_shape=len(class_names)).to(device)
FashionMNIST_Model.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=FashionMNIST_Model.parameters(),
                            lr=0.1)

start_time = timer()

EPOCHS = 10

for epoch in tqdm(range(EPOCHS)):
  train_loss, train_acc = 0, 0
  for X, y in train_dataloader:
    X, y = X.to(device), y.to(device)
    FashionMNIST_Model.train()

    # Probabilities
    y_probs = FashionMNIST_Model(X)

    # Calculate the loss
    loss = loss_fn(y_probs, y)
    train_loss += loss

    # Accuracy
    accuracy = accuracy_fn(y, y_probs.argmax(dim=1))
    train_acc += accuracy

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  FashionMNIST_Model.eval()
  with torch.inference_mode():

    test_loss, test_acc = 0, 0

    for X, y in test_dataloader:
      X, y = X.to(device), y.to(device)
      y_probs = FashionMNIST_Model(X)

      loss = loss_fn(y_probs, y)
      test_loss += loss

      acc = accuracy_fn(y, y_probs.argmax(dim=1))
      test_acc += acc

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

end_time = timer()
total_time = end_time - start_time

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=device):
  loss, acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      X, y = X.to(device), y.to(device)

      y_probs = model(X)

      loss += loss_fn(y_probs, y)
      acc += accuracy_fn(y, y_probs.argmax(dim=1))

    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name": model.__class__.__name__,
          "model_loss": loss.item(),
          "model_acc": acc}

model_results = eval_model(model=FashionMNIST_Model,
                           data_loader=test_dataloader,
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn,
                           device=device)

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):

  model.to(device)
  pred_probs=[]

  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample, dim=0).to(device)

      pred_logit = model(sample)

      pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

      pred_probs.append(pred_prob.cpu())

  return torch.stack(pred_probs)

random.seed(42)
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
  test_samples.append(sample)
  test_labels.append(label)

pred_probs = make_predictions(model=FashionMNIST_Model,
                              data=test_samples)

pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
  plt.subplot(nrows, ncols, i+1)
  plt.imshow(sample.squeeze(), cmap="gray")

  # Model predictions
  pred_label = class_names[pred_classes[i]]

  # Actual labels
  truth_label = class_names[test_labels[i]]

  title_text = f"Pred: {pred_label} | Truth: {truth_label}"

  if pred_label == truth_label:
    plt.title(title_text, fontsize=10, c="g")
  else:
    plt.title(title_text, fontsize=10, c="r")

  plt.axis(False)

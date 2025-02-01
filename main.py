import torch
from torch import nn

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from timeit import default_timer as timer

from helper_functions import accuracy_fn
from TinyVGG import TinyVGG

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

class_names = train_data.classes

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
print(total_time)

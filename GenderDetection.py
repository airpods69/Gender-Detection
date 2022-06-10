import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = ['female', 'male']

def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()

model = torch.load("./resnet18-f37072fd.pth")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
model.load_state_dict(torch.load("./FaceGenderDetectionModel.pth"))
model.to(device)


# Read the image
image = Image.open('./TestNew/jayan.jpg')

# Define a transform to convert the image to tensor
transform = transforms.Resize((224, 224))
image = transform(image)
transform = transforms.ToTensor()
image = transform(image)
transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
image = transform(image)
image = image.unsqueeze(0)

model.eval()

image = image.to(device)
outputs = model(image)
_, preds = torch.max(outputs, 1)
images = torchvision.utils.make_grid(image)
imshow(images.cpu(), title=[class_names[x] for x in preds[:1]])
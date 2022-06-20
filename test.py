import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    print("The conf value is : ", np.exp(output.data.cpu().numpy().max()))
    return index

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                     ])

to_pil = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('resnet_material_fullmodel.pth')
model.eval()


im = Image.open('/home/ubuntu/material_project/Test_Images/hickory.jpg')
index = predict_image(im)

mat_classes = ['Bricks', 'fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone', 'water', 'wood']

print("The material prediction is : ", mat_classes[index])

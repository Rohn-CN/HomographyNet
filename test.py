import argparse
from threading import main_thread
from cv2 import DescriptorMatcher
from google.protobuf import descriptor
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models


class HomoNet(nn.Module):
    def __init__(self, use_pretrained=True, feature_extract=True):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=use_pretrained)
        self.model_ft.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for para in self.model_ft.parameters():
            para.requires_grad = feature_extract
        fc_features = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(fc_features, 8)

    def forward(self, x):
        out = self.model_ft(x)
        return out


def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.image_path):
        print("image doesn't exists")
        return 0
    if not os.path.exists(args.model_path):
        print("model doesn't exists")
        return 0
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ])
    model = HomoNet()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model = model.to(device)
    image = Image.open(args.image_path)
    image = np.array(image)[..., 1:]
    image = Image.fromarray(image)
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    out = model.forward(image_tensor).to("cpu")
    out = [o.detach().numpy().item() for o in out]
    corner = np.array([[0, 0],
                       [0, 128],
                       [128, 0],
                       [128, 128]])
    corner_trans = corner+np.array(out).reshape(4, 2)
    homo = cv2.getPerspectiveTransform(np.float32(corner),np.float32(corner_trans))
    print("八参数偏移量:")
    print(list(out))
    print("单应变换：")
    print(homo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HomographyNet test")
    parser.add_argument("--image_path", help="load a image to test")
    parser.add_argument("--model_path", default=r"./model.ckpt",help="load your pretrained model")
    args = parser.parse_args()
    test(args)

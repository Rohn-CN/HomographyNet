import argparse
from cv2 import DescriptorMatcher
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models


class HomoData(Dataset):
    def __init__(self, csv_path, file_path, mode="train", valid_ratio=0.2):
        self.file_path = file_path
        self.csv_path = csv_path
        self.mode = mode

        # csv解析
        self.data_info = pd.read_csv(csv_path)
        self.data_len = len(self.data_info.index)

        self.train_len = int(self.data_len * (1 - valid_ratio))
        self.valid_len = self.data_len - self.train_len

        if mode == "train":
            self.train_image = np.array(
                self.data_info.iloc[:self.train_len, 0])
            self.train_label = np.array(
                self.data_info.iloc[:self.train_len, 1:], dtype=np.float32)
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == "valid":
            self.valid_image = np.array(
                self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.array(
                self.data_info.iloc[self.train_len:, 1:], dtype=np.float32)
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        print("finish reading the {} set,{}samples found".format(
            mode, len(self.image_arr)))

    def __getitem__(self, index):
        filename = self.image_arr[index]
        image = Image.open(os.path.join(self.file_path, filename))
        image = np.array(image)[..., 1:]
        image = Image.fromarray(image)
        if self.mode == "train":
            transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ])
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ])
        return transform(image), self.label_arr[index].reshape(8, )

    def __len__(self):
        return len(self.image_arr)


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


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = HomoNet(use_pretrained=args.use_pretrained,
                    feature_extract=args.feature_extract)
    # 查看是否需要在预训练模型基础上继续训练
    if args.pretrained == True:
        if os.path.exists(args.pretrained_model):
            model.load_state_dict(torch.load(args.pretrained_model))
        else:
            print("pretrained model doesn't exit,please check and retry")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter(args.tensorboard_path)

    train_dataset = HomoData(args.csv_file, args.file_path, "train")
    valid_dataset = HomoData(args.csv_file, args.file_path, "valid")
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size, num_workers=0, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size, num_workers=0, shuffle=True)

    best_loss = 1000
    EPOCH = args.num_epochs
    for epoch in range(EPOCH):
        model.train()
        train_loss = []
        # lr_scheduler.step()
        for batch in tqdm(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        print("Train|epoch:{:03d}/{:03d},loss = {:.5f}".format(epoch +
                                                               1, EPOCH, sum(train_loss) / len(train_loss)))

        model.eval
        valid_loss = []
        for batch in tqdm(valid_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outs = model(images)
            loss = criterion(outs, labels)
            valid_loss.append(loss)
        writer.add_scalars("loss2", {"train_loss": sum(train_loss) / len(train_loss),
                                     "valid_loss": sum(valid_loss) / len(valid_loss)}, epoch)
        print("Valid|epoch:{:03d}/{:03d},loss = {:.5f}".format(epoch +
                                                               1, EPOCH, sum(valid_loss) / len(valid_loss)))

        if sum(valid_loss) / len(valid_loss) < best_loss:
            best_loss = sum(valid_loss) / len(valid_loss)
            torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HomographyNet training")
    parser.add_argument("--file_path", default=r"./gen_dataset")
    parser.add_argument("--csv_file", default=r"./gt.csv")
    parser.add_argument("--save_path", default=r"./saved_model.ckpt")
    parser.add_argument("--use_pretrained", default=False)
    parser.add_argument("--feature_extract", default=True)
    # 超参数
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tensorboard_path", default="run/scalar_example")
    parser.add_argument("--num_epochs", default=200)
    parser.add_argument("--pretrained_model", default="./pre_res_model.ckpt")
    parser.add_argument("--pretrained", type=bool, default=False)
    args = parser.parse_args()
    train(args)

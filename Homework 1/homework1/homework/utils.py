from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np
from os import listdir
import torch

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        label_dict = dict(zip(LABEL_NAMES, range(len(LABEL_NAMES))))
        self.labels, self.files = [], []

        with open(dataset_path + '/labels.csv', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                self.labels.append(label_dict[row[1]])
                self.files.append(dataset_path + '/' + row[0])


    def __len__(self):
        """
        Your code here
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        img_data = Image.open(self.files[idx])
        img = transforms.ToTensor()(img_data)

        return img, self.labels[idx]

def load_data(dataset_path, num_workers=0, batch_size=128, shuffle=True):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

if __name__ == '__main__':
    train = SuperTuxDataset('data/train')
    print(train[0][0].shape)
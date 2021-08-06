from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np
from os import listdir
import torch

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
LABEL_NAMES_DICT = dict(zip(LABEL_NAMES,range(len(LABEL_NAMES))))

# class SuperTuxDataset(Dataset):
#     def __init__(self, dataset_path):
#         """
#         Your code here
#         Hint: Use the python csv library to parse labels.csv
#         """
#         super().__init__()
#         self.files = []
#         self.labels = []
#         with open(dataset_path + '/labels.csv','r') as f:
#             reader = csv.reader(f, delimiter=',')
#             next(reader)
#             for row in reader:
#                 self.files.append(dataset_path + '/' + row[0])
#                 self.labels.append(row[1])
#
#
#     def __len__(self):
#         """
#         Your code here
#         """
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         """
#         Your code here
#         return a tuple: img, label
#         """
#         img = Image.open(self.files[idx])
#         image_to_tensor = transforms.ToTensor()
#         img = image_to_tensor(img)
#         label = self.labels[idx]
#         return img, LABEL_NAMES_DICT[label]

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        label_dict = dict(zip(LABEL_NAMES, range(len(LABEL_NAMES))))
        self.labels = []
        self.images = []

        with open(dataset_path + '/labels.csv', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                self.labels.append(label_dict[row[1]])

                img_data = Image.open(dataset_path + '/' + row[0])
                # store loaded image
                self.images.append(transforms.ToTensor()(img_data))


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
        return self.images[idx], self.labels[idx]

def load_data(dataset_path, num_workers=0, batch_size=128, shuffle=True):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

if __name__ == '__main__':
    train = SuperTuxDataset('data/train')
    print(train[0][0].shape)
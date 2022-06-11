import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import filters

class MyDataset(Dataset):

    def samples(self, minutes):
        return minutes - ((self.window//2)*2)

    def __init__(self, dir, window, delimiter=';', device="cpu"):
        self.device = device
        self.dir = dir
        self.window = window

        self.features, self.labels = [], []

        # Можно же заменить функцией max
        max_value = -1000
        neg_bias = 1000

        with open(dir) as r_file:
            file_reader = csv.reader(r_file, delimiter=delimiter)

            for row in file_reader:
                feature = float(row[0].replace(",", "."))
                label = int(row[1])

                self.features.append(feature)
                self.labels.append(label)

                if feature > max_value:
                    max_value = feature
                if feature < neg_bias:
                    neg_bias = feature

        if neg_bias > 0:
            neg_bias = 0
        else:
            neg_bias = abs(neg_bias)
        self.features = [(neg_bias + i) / (max_value + neg_bias) for i in self.features]

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        self.features = filters.gaussian_filter(self.features, 9)
        self.features = filters.average_filter(self.features, 9)

    def __getitem__(self, index):
        features = torch.tensor(self.features[index : index+self.window], dtype=torch.float32, device=self.device)

        id = self.labels[index + self.window//2]
        label = torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32)
        label[id] = 1

        features = features.unfold(0, self.window, 1)
        label = label.unfold(0, 3, 1)

        return features, label

    def __len__(self):
        return self.samples(len(self.features))


"""
    TEST
"""
if __name__ == "__main__":

    dataset = MyDataset("..\\assets\\all.csv", 5)

    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)

    features, labels = next(iter(dataloader))

    print(features, labels)
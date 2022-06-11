import torch
import torch.nn as nn
from mydata import MyDataset
from torch.utils.data import DataLoader
from mylstm import Sequence
import trainer

device = torch.device("cuda")
lr = 0.05
epochs = 20
steps = 10
window = 1


if __name__ == "__main__":

    dataset = MyDataset("..\\assets\\1.csv", window=window, device=device)
    dataloader = DataLoader(dataset=dataset, batch_size=steps, shuffle=False)

    model = Sequence(window, 10, 3, device)

    model.to(device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainer.fit(model, criterion, optimizer, dataloader, epochs)


    dataset = MyDataset("..\\assets\\1.csv", window=window, device=device)
    dataloader = DataLoader(dataset=dataset, batch_size=steps, shuffle=False)

    trainer.test(model, dataloader)


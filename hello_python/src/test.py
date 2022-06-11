import torch
import torch.nn as nn
from mydata import MyDataset
from torch.utils.data import DataLoader
from mylstm import Sequence


device = torch.device("cuda")
lr = 0.05
epochs = 10
steps = 10
window = 5

dataset = MyDataset("..\\assets\\2.csv", window=window, device=device)
dataloader = DataLoader(dataset=dataset, batch_size=steps, shuffle=False)

total_samples = len(dataset)


model = Sequence(5, 10, 3, device)
model.to(device=device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


for epoch in range(epochs):
    print(epoch)
    loss_sum = 0
    for i, (inputs, labels) in enumerate(dataloader):
        P = model(inputs)

        Loss = criterion(labels, P)
        loss_sum += Loss.item()
        Loss.backward()

        optimizer.step()

        optimizer.zero_grad()
        model.update()
    print(loss_sum)




import matplotlib.pyplot as plt

x, y = [[],[],[]], [[],[],[]]
counter = 0
for i, (inputs, labels) in enumerate(dataloader):
    P = model(inputs)
    print(inputs.size())
    print(P.size())
    for i in range(len(P)):
        l = torch.argmax(P[i][0]).item()
        y[l].append((inputs[i][0][(window // 2)]).item())
        x[l].append(counter)
        counter += 1

fig, ax = plt.subplots()
ax.scatter(x[0], y[0], label="UP")
ax.scatter(x[1], y[1], label="DOWN")
ax.scatter(x[2], y[2], label="OTHER")
ax.set_xlabel("minutes")
ax.set_ylabel("values")
ax.legend(loc="best")

plt.show()
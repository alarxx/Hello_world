import torch
import matplotlib.pyplot as plt


def fit(model, criterion, optimizer, dataloader, epochs):
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
        print(loss_sum)


def test(model, dataloader):
    x, y = [[], [], []], [[], [], []]
    counter = 0
    for i, (inputs, labels) in enumerate(dataloader):
        P = model(inputs)
        for i in range(len(P)):
            l = torch.argmax(P[i][0]).item()
            y[l].append((inputs[i][0][(len(inputs[i][0]) // 2)]).item())
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
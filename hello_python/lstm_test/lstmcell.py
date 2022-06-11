import torch
import torch.nn as nn

if __name__ == "__main__":
    rnn = nn.LSTMCell(10, 3) # (input_size, hidden_size)
    input = torch.randn(2, 1, 10) # (time_steps, batch, input_size)
    hx = torch.randn(1, 3) # (batch, hidden_size)
    cx = torch.randn(1, 3)

    print(input)
    print(hx)
    print(cx)

    output = []
    print(input.size())
    for i in range(input.size()[0]):
        print(input[i])
        hx, cx = rnn(input[i], (hx, cx))
        output.append(hx)
    output = torch.stack(output, dim=0)

    print(input)
    print(hx)
    print(cx)


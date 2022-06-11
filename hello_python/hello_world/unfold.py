import torch

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3], [1, 2, 3]])
    print(x)

    x = x.unfold(1, 3, 1)
    print(x)
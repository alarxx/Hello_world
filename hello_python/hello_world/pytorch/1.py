import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

train_data_path = "./dataset_9min/train"
val_data_path = "./dataset_9min/val"
test_data_path = "./dataset_9min/test"

transform = transforms.Compose([ 
	transforms.Resize(28), 
	transforms.ToTensor(), 
	transforms.Normalize(mean=[0.5, 0.5, 0.5], 
		std=[0.5, 0.5, 0.5])])

train_data = torchvision.datasets.ImageFolder(
	root=train_data_path, 
	transform=transforms)

test_data = torchvision.datasets.ImageFolder(
	root=test_data_path, 
	transform=transforms)


labels_map = {
    0: "B",
    1: "V",
}


figure = plt.figure(figsize=(8, 8))

cols, rows = 28, 28

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


"""
val_data = torchvision.datasets.ImageFolder(
	root=val_data_path, 
	transform=transforms)

batch_size = 8

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size)
val_data_loader = torch.utils.data.DataLoader(train_data, batch_size)
test_data_loader = torch.utils.data.DataLoader(train_data, batch_size)
"""
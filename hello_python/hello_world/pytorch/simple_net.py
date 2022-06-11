import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 84)
		self.fc2 = nn.Linear(84, 42)
		self.fc3 = nn.Linear(42, 2)

	def forward(self, x):
		x = x.view(-1, 784)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

simpleNet = SimpleNet()
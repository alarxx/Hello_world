import torch
import torch.nn as nn

dtype = torch.float
device = torch.device("cpu") 

I = torch.tensor([[1, 2, 3]], dtype=dtype, device=device)
T = torch.tensor([[2, 4]], dtype=dtype, device=device)

class LinearRegression(nn.Module):
	def __init__(self, nIn, nOut):
		super().__init__()
		self.lin = nn.Linear(nIn, nOut)

	def forward(self, x):
		return self.lin(x)

model = LinearRegression(3, 2)
model.to(device=device)

lr = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
	P = model(I)

	Loss = loss(T, P)

	Loss.backward()
		
	optimizer.step()
	optimizer.zero_grad()

	if epoch % 1 == 0: 
		print(f"epoch {epoch}: loss={Loss}")
		print("---P---\n", P)


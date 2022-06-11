import torch

def benchmark(f):
	import time
	def wrapper():
		start = time.time()
		f()
		end = time.time()
		print("[*] время выполнение: {} секунд".format(end-start))
	return wrapper

dtype = torch.float32
device = torch.device("cpu") 

I = torch.tensor([[1, 2, 3]], dtype=dtype, device=device)

T = torch.tensor([[2, 4]], dtype=dtype, device=device)

W = torch.tensor([
	[1, 1],
	[1, 1],
	[1, 1]], dtype=dtype, device=device, requires_grad=True)

#returns Prediction Y
def forward(I):
	return I.matmul(W)

def loss(T, P):
	return (T - P).pow(2).mean()


lr = 0.01

@benchmark
def train():
	for epoch in range(100000):
		global W

		P = forward(I)

		Loss = loss(T, P)
		Loss.backward()
		
		with torch.no_grad():
			W -= lr * W.grad
			W.grad.zero_()

		if epoch % 1000 == 0: 
			print(f"epoch {epoch}: loss={Loss}")
			print("---P---\n", P)

train()

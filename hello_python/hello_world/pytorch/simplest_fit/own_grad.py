import torch

def benchmark(f):
	import time
	def wrapper():
		start = time.time()
		f()
		end = time.time()
		print("[*] время выполнение: {} секунд".format(end-start))
	return wrapper



dtype = torch.float
device = torch.device("cpu") 

I = torch.tensor([[1, 2, 3]], dtype=dtype, device=device)

T = torch.tensor([[2, 4]], dtype=dtype, device=device)

W = torch.tensor([
	[1, 1],
	[1, 1],
	[1, 1]], dtype=dtype, device=device)

#returns Prediction Y
def forward(I):
	return I.matmul(W)

def loss(T, P):
	return (T - P).pow(2).mean()

def gradient(I, T, P):
	return I.reshape(3, 1).matmul(-2*(T-P))


lr = 0.01
@benchmark
def train():
	for epoch in range(100000):
		global W
		P = forward(I)

		Loss = loss(T, P)

		dW = gradient(I, T, P)

		W -= lr * dW

		if epoch % 1 == 0: 
			print(f"epoch {epoch+1}: loss={Loss}")
		print("---P---\n", P)

train()
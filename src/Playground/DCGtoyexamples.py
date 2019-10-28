import torch

pattern = torch.tensor([2., 3.])
fc1 = torch.nn.Linear(2, 1, bias=False)
fc2 = torch.nn.Linear(2, 2, bias=False)
fc3 = torch.nn.Linear(2, 1, bias=False)

fc1.weight.data = torch.Tensor([[1., 2.]])
fc2.weight.data = torch.Tensor([[3.], [4.]])
fc3.weight.data = torch.Tensor([[5., 6.]])

hout1 = fc1(pattern)
hout2 = fc2(hout1)
out = fc3(hout2)
out.backward()
print(out)
pass
import faulthandler
from toygrad_cpu import Tensor, Shape, TensorGraph

faulthandler.enable()
t1 = Tensor.randn([2, 3, 4])
t2 = Tensor.randn([2, 3, 4])
t3 = t1.max(1)
t4 = t3.sum()
graph = TensorGraph.from_tensor(t4)
graph.forward()
graph.backward()
print(t1)
print(t3)
print(t1.grad())
from toygrad_cpu import Tensor, Shape, TensorGraph

t1 = Tensor.randn([2, 3, 4])
t2 = Tensor.randn([2, 3, 4])
t3 = t1 + t2
graph = TensorGraph.from_tensor(t3)
graph.forward()
print(t1)
print(t2)
print(t3)
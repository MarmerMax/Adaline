import InputReader
import numpy as np

dataset = InputReader.get_dataset()

x = dataset[0]
# dataset_np = np.array(x)
# x = np.empty((0, 13), float)

for line in dataset:
    b = np.array(dataset[1])
    dataset_np = np.vstack((dataset_np, b))

print(dataset_np)
print(len(dataset_np))
# print(len(dataset))
# print(len(dataset[0]))

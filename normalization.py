import math
import numpy as np


# Step to softmax Input -> exponentiate -> normalize -> output

E = math.e

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)

# exp_values = []

# for i in layer_outputs:
#     exp_values.append(E**i)

# normalize

norm_base = sum(exp_values)
norm_values = exp_values / np.sum(exp_values)

# norm_values = []

# for value in exp_values:
#     norm_values.append(value / norm_base)


print(exp_values)
print(norm_values)
print(sum(norm_values))

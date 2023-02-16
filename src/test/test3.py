import numpy as np
table = np.zeros((16, 1))
mask = np.ones((16, 1))
evaluations = table / mask
print(np.mean(evaluations[1:...]))

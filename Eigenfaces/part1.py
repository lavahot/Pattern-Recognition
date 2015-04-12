from train import *
from test import *
import matplotlib.pyplot as plt

VAR_KEEPS = [0.8, 0.9, 0.95]
perfs = []

for varKeep in VAR_KEEPS:
	print('\n\nTraining', varKeep)
	train(varKeep)
	print('\n\nTesting')
	perfs.append(test())


plt.xlabel('Performance')
plt.ylabel('Rank')
plt.title('Algorithm Performance')
plt.xlim([1.0, MAX_RANK])
plt.ylim([0.0, 1.0])
for i in range(len(VAR_KEEPS)):
	perf = perfs[i]
	plt.plot(range(1, MAX_RANK+1), perf, label=str(VAR_KEEPS[i]))
plt.legend()
plt.show()

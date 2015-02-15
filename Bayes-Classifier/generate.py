import sys, random, io
import numpy as np
from numpy import ndarray

def showHelp():
	print('fname, nSamples, nDims mu1 mu2 ... mu_d sigm1 sigm2 ... sigmd')

if __name__ == '__main__':
	args = sys.argv[1:]
	if len(args) == 0:
		showHelp()
	else:
		args.reverse() # Allows stack like usage
		fname = args.pop()
		nSamples = int(args.pop())
		nDims = int(args.pop())
		arr = np.empty((nSamples, nDims))
		mus = []
		sigms = []
		for i in range(nDims):
			mus.append(float(args.pop()))
		for i in range(nDims):
			sigms.append(float(args.pop()))
		for i in range(nSamples):
			for j in range(nDims):
				arr[i, j] = random.gauss(mus[j], sigms[j])
		fp = open(fname, 'wb')
		np.savetxt(fp, arr)
		fp.close()
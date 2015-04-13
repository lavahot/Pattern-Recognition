import glob, PIL.Image
from pca import *
from image_manipulation import *
import numpy as np
import matplotlib.pyplot as plt

# Load training data
print('Loading training data.')
trainingFiles = glob.glob('fa2_H/*.pgm')
nTraining = len(trainingFiles)
h, w = readImage(trainingFiles[0]).shape
nDims = h * w
trainingVects = np.empty([nDims, nTraining], dtype='float')
for i in range(nTraining):
	img = readImage(trainingFiles[i])
	x = vectorizeImage(img)
	trainingVects[:, i] = x
def getId(fname):
	for delim in ['/', '\\']:
		fname = fname.rsplit(delim, 2)[-1]
	return int(fname.split('_', 2)[0])
trainingIds = set(map(getId, trainingFiles))


# PCA
print('Obtaining PCA space.')
pca = Pca(trainingVects, 0.95)

# Load testing data
print('Loading testing data.')
testingFiles = glob.glob('fb_H/*.pgm')
nTesting = len(testingFiles)
testingDists = np.empty(nTesting)
for i in range(nTesting):
	fname = testingFiles[i]
	img = readImage(fname)
	x = vectorizeImage(img)
	testingDists[i] = pca.getReconstructionError(x)
testingIds = list(map(getId, testingFiles))
testingIds = np.array(testingIds, dtype='int')

# Get Positive and Negative Count
nPositives = 0.0
nNegatives = 0.0
for ID in testingIds:
	if ID in trainingIds:
		nPositives += 1.0
	else:
		nNegatives += 1.0

# Perform test
print('Performing tests')
x = np.linspace(testingDists.min(), testingDists.max(), 100000)
tpr = []
fpr = []
for thresh in x:
	truePositives = 0
	falsePositives = 0
	for i in range(nTesting):
		predict = (testingDists[i] < thresh)
		truth = (testingIds[i] in trainingIds)
		if predict and truth:
			truePositives += 1
		elif predict and not truth:
			falsePositives += 1
	tpr.append(truePositives/nPositives)
	fpr.append(falsePositives/nNegatives)
print(nPositives, nNegatives)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Intrusion Detection')
plt.plot(fpr, tpr)
plt.show()

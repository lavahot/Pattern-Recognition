from pca import *
from image_manipulation import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Get training
trainingFiles = glob.glob('fa_H/*.pgm')

# Get the ids
def getId(fname):
	for delim in ['/', '\\']:
		fname = fname.rsplit(delim, 2)[-1]
	return int(fname.split('_', 2)[0])
trainingIds = list(map(getId, trainingFiles))
trainingIds = np.array(trainingIds, dtype='int')

# Get dimension info
h, w = readImage(trainingFiles[0]).shape
d = h * w
n = len(trainingFiles)

# Load testing images
print('Loading training data')
trainingVects = np.empty((d, n), dtype='float')
for i in range(n):
	img = readImage(trainingFiles[i])
	x = vectorizeImage(img)
	trainingVects[:, i] = x

# Apply PCA
print('Obtaining eigenfaces')
pca = Pca(trainingVects, 0.8)

# Test reprojection
#x = pca.getReconstruction(trainingVects[:, 0])
#plt.imshow(x.reshape([h, w]), cmap='gray')
#plt.show()

# Write images to file
print('training/mean.png')
print('training/eigenFace(n).png')
writeImage('training/mean.png', devectorizeImage(pca.meanVect, w, h))
for i in range(10):
	eigenFace = devectorizeImage(pca.eigenvectors[:, i].copy(), w, h)
	eigenFace -= eigenFace.min()
	eigenFace *= 255.0 / (eigenFace.max())
	destFile = 'training/eigenFace{n}.png'.format(n=i)
	writeImage(destFile,
	           eigenFace)

#for i in range(1, 11):
#	eigenFace = devectorizeImage(pca.eigenvectors[:, -i].copy(), w, h)
#	eigenFace -= eigenFace.min()
#	eigenFace *= 255.0 / (eigenFace.max())
#	destFile = 'training/worstEigenFace{n}.png'.format(n=i-1)
#	writeImage(destFile,
#	           eigenFace)

# Store items to file
np.savetxt('training/eigenvalues.txt', pca.eigenvalues)
np.savetxt('training/eigenvectors.txt', pca.eigenvectors)
np.savetxt('training/ids.txt', trainingIds)

# Project training images to eigenspace
print('Projecting faces')
k = pca.k
trainingProjection = np.empty([k, n])
for i in range(n):
	x = trainingVects[:, i].copy()
	y = pca.project(x)
	trainingProjection[:, i] = y
np.savetxt('training/projection.txt', trainingProjection)

pickle.dump(pca, open('training/pca.pkl', 'wb'))
from pca import *
from image_manipulation import *
import glob
import matplotlib.pyplot as plt
import numpy as np

# Get training and testing images
trainingFiles = glob.glob('fa_L/*.pgm')

# Get the ids
def getId(fname):
	for delim in ['/', '\\']:
		fname = fname.rsplit(delim, 2)[-1]
	return int(fname.split('_', 2)[0])
trainingIds = list(map(getId, trainingFiles))

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
pca = Pca(trainingVects, 1.0)

# Test reprojection
x = pca.getReconstruction(trainingVects[:, 0])
plt.imshow(x.reshape([h, w]), cmap='gray')
plt.show()

# Write images to file
print('tests/mean.png')
print('tests/eigenFace(n).png')
writeImage('tests/mean.png', devectorizeImage(pca.meanVect, w, h))
for i in range(10):
	eigenFace = devectorizeImage(pca.eigenvectors[:, i].copy(), w, h)
	eigenFace -= eigenFace.min()
	eigenFace *= 255.0 / (eigenFace.max())
	destFile = 'tests/eigenFace{n}.png'.format(n=i)
	writeImage(destFile,
	           eigenFace)

# Store items to file
print('tests/eigenvalues.txt')
np.savetxt('tests/eigenvalues.txt', pca.eigenvalues)
np.savetxt('tests/eigenvectors.txt', pca.eigenvectors)
np.savetxt('tests/ids.txt', np.array(trainingIds, dtype='int'))

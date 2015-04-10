from pca import *
from image_manipulation import *
import glob
import matplotlib.pyplot as plt
import numpy as np

# Get training and testing images
trainingFiles = glob.glob('fa_H/*.pgm')
testingFiles = glob.glob('fb_H/*.pgm')

# Get dimension info
h, w = readImage(trainingFiles[0]).shape
d = h * w
n = len(trainingFiles)

# Load images
print('Loading training data')
trainingVects = np.empty((d, n), dtype='float')
for i in range(n):
	img = readImage(trainingFiles[i])
	x = vectorizeImage(img)
	trainingVects[:,i] = x

# print "Here are the training vectors:\n", trainingVects
# PCA
pca = Pca(trainingVects, 0.95)

# Test mean face
print "Showing mean face. Rawr!"
meanFace = devectorizeImage(pca.meanVect, w, h)
plt.title('Mean Face')
plt.imshow(meanFace, cmap='gray')
plt.show()

# Test first eigenface
print "Showing " + str(pca.k) + " Eigenfaces out of " + str(len(pca.eigenvalues)) + "."
for i in range(pca.k):
	eigenFace = devectorizeImage(pca.eigenvectors[:, pca.eigensort[i]], w, h)
	plt.title('Eigen Face ' + str(i+1))
	plt.imshow(eigenFace, cmap='gray')
	plt.show()

from pca import *
from image_manipulation import *
import glob
import matplotlib.pyplot as plt
import numpy as np

# Get training and testing images
trainingFiles = glob.glob('fa_L/*.pgm')
testingFiles = glob.glob('fb_L/*.pgm')

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
	trainingVects[:, i] = x

# PCA
pca = Pca(trainingVects, 1.0)

# Test mean face
meanFace = devectorizeImage(pca.meanVect, w, h)
plt.title('Mean Face')
plt.imshow(meanFace, cmap='gray')
plt.show()

# Test first eigenface
eigenFace = devectorizeImage(pca.eigenVects[:, 0], w, h)
plt.title('Eigen Face')
plt.imshow(eigenFace, cmap='gray')
plt.show()

from pca import *
from image_manipulation import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

# Get training and testing images
res = str(raw_input("Please enter the resolution of faces you'd like to train for (H/l): ") or 'H').upper()
trainingFiles = glob.glob('fa_'+ res + '/*.pgm')
testingFiles = glob.glob('fb_' + res + '/*.pgm')

# Get dimension info
h, w = readImage(trainingFiles[0]).shape
d = h * w
n = len(trainingFiles)
k = float(raw_input("Enter the percent of information you wish to keep (0.95): ") or "0.95")
# Load images
print('Loading training data')
trainingVects = np.empty((d, n), dtype='float')
for i in range(n):
	img = readImage(trainingFiles[i])
	x = vectorizeImage(img)
	trainingVects[:,i] = x

# print "Here are the training vectors:\n", trainingVects
# PCA
pca = Pca(trainingVects, k)

# Make eigenface directory if it doesn't exist.
if not os.path.exists("ef_" + res):
	os.makedirs("ef_" + res)

# Display and save mean face
print "Showing mean face. Rawr!"
meanFace = devectorizeImage(pca.meanVect, w, h)
plt.title('Mean Face')
plt.imshow(meanFace, cmap='gray')
plt.show()
writeImage("ef_" + res + "/mean.pgm", meanFace)

# Display and save K eigenfaces
print "Showing " + str(pca.k) + " Eigenfaces out of " + str(len(pca.eigenvalues)) + "."
for i in range(pca.k):
	eigenFace = devectorizeImage(pca.eigenvectors[:, pca.eigensort[i]], w, h)
	plt.title('Eigen Face ' + str(i+1))
	plt.imshow(eigenFace, cmap='gray')
	plt.show()
	writeImage("ef_" + res + "/ef" + str(i) + ".pgm", eigenFace)

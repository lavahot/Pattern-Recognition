from pca import *
from image_manipulation import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

# Select training or testing mode
mode = str(raw_input("Please select tRaining or tEsting mode (R/E): ")).upper()
# Get training and testing images
res = str(raw_input("Please enter the resolution of faces you'd like use (H/l): ") or 'H').upper()
trainingFiles = glob.glob('fa_'+ res + '/*.pgm')
testingFiles = glob.glob('fb_' + res + '/*.pgm')

if mode == 'R':

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

	# Save eigenvalues
	evf = open("ef_" + res + "/eigvals.txt", 'w')
	for i in pca.eigenvalues[pca.eigensort[:pca.k]]:		
		evf.write("%s\n" % i)
	evf.close()

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

	# Calculate the coefficeints of projection for the test set.
	copf = "ef_" + res + "/cop.txt"
	cop = np.empty(pca.dim)
	for i in trainingVects[:, pca.eigensort]:
		cop[i] = pca.project(i)	
	np.savetxt(copf, cop)

elif mode == 'E':
	# Read in values from files
	pca = Pca.load("ef_" + res)
	n = len(testingFiles)
	with open("ef_" + res + "/cop.txt", 'r+') as copf:
		cop = map(float, copf)
	# Generate projection coeffs for test files
	for i in testingVects: 
		copv = pca.project(i)
		errc = 1000
		for j in cop:
			if (copv-j < errc):
				errc = abs(copv-j)
				copb = j
	

else :
	print "Your selection was invalid."


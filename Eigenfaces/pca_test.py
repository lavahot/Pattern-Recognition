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
	for i in pca.eigenvalues[:pca.k]:		
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
	print "Showing " + str(pca.k) + " eigenfaces out of " + str(len(pca.eigenvalues)) + "."
	for i in range(pca.k):
		eigenFace = devectorizeImage(pca.eigenvectors[:, i], w, h)
		plt.title('Eigen Face ' + str(i+1))
		plt.imshow(eigenFace, cmap='gray')
		plt.show()
		writeImage("ef_" + res + "/ef" + str(i) + ".pgm", eigenFace)

	# Calculate the coefficeints of projection for the test set.
	print "Saving projection coefficients."
	copf = "ef_" + res + "/cop.txt"
	cop = []
	for i in trainingVects.T:
		# print i.shape
		cop.append(pca.project(i))	
	np.savetxt(copf, np.vstack(cop))

	# For experiment A1 save the mean face, and the top and bottom 10 eigenfaces.

	# For experiment A2 get eigen faces for 80% information retention.
	# Get eigen-coefficient vectors for all testing and training images
	# get mahalanobis distance for every pair of training and query images.
	# chose top N face gallery images having highest similarity score with the query face.
	# if query image is among the most N most similar faces, it is considered a correct match. What does this mean? Why would I match one face from one data set with the same face from the same dataset?
	# count the number of correct matches and divide it by the total number of images in the test set to report the id accuracy.
	# Draw the Cumulative Match Characteristic curve against N varying 1 to 50

	# A3: Assuming N=1 show 3 query images which are correctly matched,
	# along with the corresponding best matched samples.

	# A4: Assuming N=1, show 3 query images which are incorrectly matched,
	# along with the corresponding mismatched samples.

	# Repeat A2 - A4 by keeping top eigenvectors corresponding to 90 and 95% of the information in the data. Plot the CMC curves on the same graph for comparison purposes. If there are significant differences in terms of identification accuracy in a2 and a5, try to explain why. If there are no significant differences, explain why too. 


elif mode == 'E':
	# Read in values from files
	pca = Pca.load("ef_" + res)
	n = len(testingFiles)
	cop = np.loadtxt("ef_" + res + "/cop.txt")
	# Generate projection coeffs for test files
	for i in testingVects.T: 
		copv = pca.project(i)
		errc = 1000
		for j in cop:
			if (copv-j < errc):
				errc = abs(copv-j)
				copb = j


	

else :
	print "Your selection was invalid."


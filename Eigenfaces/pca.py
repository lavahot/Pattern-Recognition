import numpy as np
import numpy.linalg as la
from image_manipulation import *

# TODO:
#	Pca.__init__
#	Pca.project
#	Pca.reproject

class Pca:
	"""Build a space based on the principle components of some data.
	Class Members
	-------------
	meanVect - numpy array/vector
		Mean vector of the training data.
	eigenvectors - 2D numpy array
		Eigenvectors that represent the principle components.
	eigenvalues - numpy array
		The eigenvalues corresponding to the eigenvectors.
	"""

	def __init__(self, training=np.zeros((1,1)), keep=0.9, loadfolder=""):
		"""Initialize a PCA space based on the training data.
		Parameters
		----------
		training - 2D numpy array
			Training data in the shape of [feature, sample].
		keep - Clamped Float range=[0.0, 1.0]
			The amount of information/variance to keep.
		"""
		if len(training[0]) > 1:
			# Get dimensionality
			self.dim = len(training)
			# Step 1: Get mean sample
			print "Getting mean vector."
			self.meanVect = np.zeros(self.dim)
			for d in range(self.dim):
				for i in range(len(training[0])):
					self.meanVect[d] += float(training[d, i])
				self.meanVect[d] /= float(len(training[:, 0]))

			# Step 2: get distance from mean. 
			print "Getting mean distance matrix."
			self.theta = np.zeros((self.dim, len(training[0])))
			for i in range(len(training[0])):
				self.theta[:, i] = training[:, i] - self.meanVect

			# print "This is matrix A:\n", self.theta
			# print self.theta.size, training.size

			# Step 3: get sample covariance matrix, C
			print "Getting mean distance matrix covariance."
			self.C = self.theta.dot(self.theta.transpose()) / len(training[0])

			# Step 4: get sorted eigenvalues of C
			# Step 5: get eigenvectors of C
			print "Getting eigenvalues and eigenvectors."
			self.eigenvalues, self.eigenvectors = la.eigh(self.C)
			print "Sorting eigenvalues."
			self.eigensort = self.eigenvalues.argsort()[::-1]
			
			# Step 6: Reduce dimensionality by keeping only the largest eigenvalues and corresponding eigenvectors.
			# self.besteigen = self.eigensort[math.floor(-keep*len(training))]
			
			# Find K
			print "Finding best eigenvalues."
			sumk = self.eigenvalues.sum()
			for i in range(len(training[0])):
				if self.eigenvalues[self.eigensort[:i]].sum() / sumk > keep:
					self.k = i
					break
		else:
			with open(loadfolder + "/eigvals.txt") as evf:
				self.eigenvalues = map(float, evf)
			self.k = sum(1 for line in open(loadfolder + "/eigvals.txt"))
			self.eigenvectors = []
			for i in range(self.k):
				self.eigenvectors[i] = readImage(loadfolder + "/ef" + i +".pgm")
			self.meanVect = readImage(loadfolder + "/mean.pgm")
			self.eigensort = np.arange(self.k)


	@classmethod
	def load(cls, loadfolder):
		return cls(loadfolder)

	def project(self, x):
		"""Find the projection of x onto the PCA space.
		Parameters
		----------
		x - Numpy vector/array
			Input feature vector.

		Returns
		-------
		y - Numpy vector/array
			Projection of x onto the PCA space.
		"""
		return self.eigenvectors[self.eigensort[:self.k]].dot(np.vstack(x - self.meanVect))
		

	def reproject(self, y):
		"""Find the reprojection of y from the PCA space.
		Parameters
		----------
		y - Numpy vector/array
			Input feature vector from the PCA space.

		Returns
		-------
		x - Numpy vector/array
			Reconstruction based on y.
		"""
		return np.cross(self.project(x),self.eigenvectors[self.eigensort[:self.k]]) + self.meanVect 

	def getMahalanobisDist(self, x1, x2):
		"""Find the mahalanobis distance between x1 and x2 in the pca space.
		Parameters
		----------
		x1 - Numpy vector/array
			Point to compare against.
		x2 - Numpy vector/array
			Point to compare against.

		Returns
		-------
		dist - Float
			The mahalanobis distance between x1 and x2.
		"""
		diff = x1 - x2
		diff /= self.eigenvalues
		return la.norm(diff)

	def getReconstruction(self, x):
		"""Projects x onto the PCA space and gets the reconstruction.
		Parameters
		----------
		x - Numpy vector/array
			Input feature vector.

		Returns
		-------
		y - Numpy vector/array
			Output feature vector.
		"""
		return self.reproject(self.project(x))
	
	def getReconstructionError(self, x):
		"""Finds the reconstruction error caused by projecting x onto the PCA space.
		Parameters
		----------
		x - Numpy vector/array
			Input feature vector.

		Returns
		-------
		e - Float
			Reconstruction error
		"""
		return la.norm(x - self.getReconstruction(x))

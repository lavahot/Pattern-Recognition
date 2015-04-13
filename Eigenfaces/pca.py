import numpy as np
import numpy.linalg as la

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
	k - integer
		Number of eigenvectors kept.
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
			nFeatures = training.shape[0]
			nSamples = training.shape[1]
			self.meanVect = np.empty(nFeatures)
			for d in range(nFeatures):
				self.meanVect[d] = training[d].mean()

			# Step 2: get distance from mean.
			print "Getting mean distance matrix." 
			theta = training.copy()
			for d in range(nFeatures):
				theta[d] -= self.meanVect[d]

			# Step 3: get sample covariance matrix, C
			print "Getting mean distance matrix covariance."
			C = np.dot(theta.transpose(), theta) / nSamples

			# Step 4: get sorted eigenvalues of C
			# Step 5: get eigenvectors of C
			print "Getting eigenvalues and eigenvectors."
			self.eigenvalues, self.eigenvectors = la.eigh(C)
			print "Sorting eigenvalues."		
			self.eigenvectors = np.dot(theta, self.eigenvectors)
			self.eigenvalues = self.eigenvalues[::-1].copy()
			self.eigenvectors = self.eigenvectors[:, ::-1].copy()
			nComponents = self.eigenvalues.shape[0]
			# Normalize to unit length 1
			for i in range(nComponents):
				self.eigenvectors[:, i] /= la.norm(self.eigenvectors[:, i])

			# Step 6: Reduce dimensionality by keeping only the largest eigenvalues and corresponding eigenvectors.
			# Find K
			totSum = self.eigenvalues.sum()
			sumK = 0.0
			k = 0
			for i in range(nComponents):
				sumK += self.eigenvalues[i]
				k += 1
				if sumK / totSum >= keep:
					break
			self.k = k
			self.eigenvalues = self.eigenvalues[:k].copy()
			self.eigenvectors = self.eigenvectors[:, :k].copy()
	else:
			with open(loadfolder + "/eigvals.txt") as evf:
				self.eigenvalues = map(float, evf)
			self.k = sum(1 for line in open(loadfolder + "/eigvals.txt"))
			self.eigenvectors = []
			for i in range(self.k):
				self.eigenvectors.append(readImage(loadfolder + "/ef" + str(i) +".pgm"))
			self.meanVect = readImage(loadfolder + "/mean.pgm")
			self.eigensort = np.arange(self.k)

	@classmethod
	def load(cls, loadfolder):
		return cls(loadfolder=loadfolder)

	def project(self, x, eignum=None):
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
		if eignum == None:
			eignum = self.k
		return self.eigenvectors[:, :eignum].T.dot(np.vstack(x - self.meanVect))
		

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
		return np.dot(self.eigenvectors, y)+self.meanVect

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
		diff = diff ** 2
		diff /= self.eigenvalues
		return math.sqrt(diff.sum())

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

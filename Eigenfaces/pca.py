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
	"""

	def __init__(self, training, keep=0.9):
		"""Initialize a PCA space based on the training data.
		Parameters
		----------
		training - 2D numpy array
			Training data in the shape of [feature, sample].
		keep - Clamped Float range=[0.0, 1.0]
			The amount of information/variance to keep.
		"""
		# Get dimensionality
		nFeatures = training.shape[0]
		nSamples = training.shape[1]
		self.meanVect = np.empty(nFeatures)
		for d in range(nFeatures):
			self.meanVect[d] = training[d].mean()

		# Step 2: get distance from mean. 
		theta = training.copy()
		for d in range(nFeatures):
			theta[d] -= self.meanVect[d]

		# Step 3: get sample covariance matrix, C
		C = theta.dot(theta.transpose()) / nSamples

		# Step 4: get sorted eigenvalues of C
		# Step 5: get eigenvectors of C
		eigenvalues, eigenvectors = la.eigh(C)
		self.eigenvalues = np.empty(eigenvalues.shape)
		self.eigenvectors = np.empty(eigenvectors.shape)
		nComponents = self.eigenvalues.shape[0]
		for i in range(nComponents//2):
			j = nComponents - 1 - i
			self.eigenvalues[i] = eigenvalues[j]
			self.eigenvalues[j] = eigenvalues[i]
			self.eigenvectors[:, i] = eigenvectors[:, j]
			self.eigenvectors[:, j] = eigenvectors[:, i]

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
		self.eigenvalues = self.eigenvalues[:k].copy()
		self.eigenvectors = self.eigenvectors[:, :k].copy()
		print(self.eigenvalues)
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
		return np.dot(x, self.eigenvectors)

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
		return np.dot(self.eigenvectors, y)

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

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

	def __init__(self, training, keep=1.0):
		"""Initialize a PCA space based on the training data.
		Parameters
		----------
		training - 2D numpy array
			Training data in the shape of [feature, sample].
		keep - Clamped Float range=[0.0, 1.0]
			The amount of information/variance to keep.
		"""
		# Get dimensionality
		self.dim = len(training[0])
		# Step 1: Get mean sample
		self.meansample = np.zeros(d)
		for d in range(self.dim):
			for i in len(training):
				self.meansample[d] += float(training[d][i])
			self.meansample[d] /= float(len(training))

		# Step 2: get distance from mean. 
		self.theta = np.zeros(len(training),self.dim)
		for i in range(len(training)):
			self.theta[i] = training[i] - self.meansample

		# Step 3: get sample covariance matrix, C
		self.C = self.theta.dot(self.theta.transpose()) / len(training)

		# Step 4: get sorted eigenvalues of C
		# Step 5: get eigenvectors of C
		self.eigenvalues, self.eigenvectors = la.eigh(self.theta)
		self.eigensort = eigenvectors.argsort()
		


	
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
		pass

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
		pass

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

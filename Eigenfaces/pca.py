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
	def __init__(self, training, keep):
		"""Initialize a PCA space based on the training data.
		Parameters
		----------
		training - 2D numpy array
			Training data in the shape of [feature, sample].
		keep - Clamped Float range=[0.0, 1.0]
			The amount of information/variance to keep.
		"""
		pass
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

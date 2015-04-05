import PIL.Image
import numpy as np

def readImage(fname, dtype='float'):
	"""Reads an image from a file.

	Parameters
	----------
	fname - String
		File to read image from.
	dtype - Numpy datatype
		Datatype for array. Float is recommended for image manipulation.

	Returns
	-------
	img - 2D numpy array
		The image that is read from the file.
	"""
	return np.array(PIL.Image.open(fname), dtype=dtype)

def writeImage(fname, img):
	"""Writes an image to file.
	Parameters
	----------
	fname - String
		File to write image to.
	img - 2D numpy array
		The image should contain pixels in range [0, 255].

	Returns
	-------
	"""
	img = np.array(img, dtype='uint8')
	img = PIL.Image.fromarray(img)
	img.save(fname)

def vectorizeImage(img):
	"""Returns a vectorized version of img.
	Parameters
	----------
	img - 2D or 3D numpy array
		The image

	Returns
	-------
	x - numpy vector/array
		A copy of image with the x and y dimensions concatinated.
	"""
	return img.reshape(np.prod(img.shape)).copy()

def devectorizeImage(x, width, height):
	"""Returns a 2D(grayscale) or 3D(color) image from the vector x.
	Parameters
	----------
	x - 1D numpy array/vector
		Vectorized image
	width - Integer
		The width of the image
	height - Integer
		The height of the image

	Returns
	-------
	img - 2D or 3D numpy array
		A grayscale or color version of the image represented by vector x.
	"""
	totSize = np.prod(img.shape)
	nChannels = totSize / (width * height)
	if nChannels == 1:
		return img.reshape((height, width)).copy()
	else:
		return img.reshape((height, width, nChannels)).copy()


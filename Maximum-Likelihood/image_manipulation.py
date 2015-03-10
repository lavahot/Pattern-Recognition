import PIL.Image
import numpy as np
from numpy import ma

def readImg(fname):
	'''
	Returns as a numpy array of floats between 0 and 255.
	'''
	return np.array(PIL.Image.open(fname), dtype='float')

def writeImg(img, fname):
	''' Writes an image to a file.
	Keyword arguments:
	img -- A numpy array which holds the image. The values should be between 0 and 255.
	fname -- The output filename.
	'''
	img = PIL.Image.fromarray(img.astype('uint8'))
	img.save(fname)

def rgbToGray(img):
	gray = img[:, :, 0]/3 + img[:, :, 1]/3 + img[:, :, 2]/3
	return gray

def getPixels(img, mask):
	''' Returns an array of rgb pixels in the form (sample, RGB components).
	Keyword arguments:
	img -- A numpy array which holds the image.
	mask -- A numpy array in which values non-zero values are kept.
	'''
	# Convert image to vector of pixels
	if len(mask.shape) > 2:
		mask = rgbToGray(mask)
	mask = mask.reshape(np.prod(mask.shape))
	mask = (mask==0) # 0's are masked out
	img = img.reshape((np.prod(img.shape[:2]), 3))
	# Mask each of the components
	r, g, b = img[:, 0], img[:, 1], img[:, 2]
	r = ma.array(r, mask=mask).compressed()
	g = ma.array(g, mask=mask).compressed()
	b = ma.array(b, mask=mask).compressed()
	# Combine rgb
	pixels = np.empty((len(r), 3))
	pixels[:, 0] = r
	pixels[:, 1] = g
	pixels[:, 2] = b
	#
	return pixels

def RGBPixelsToYCbCr(pixels):
	''' Returns an array of YCbCr pixels in the form (sample, YCbCr components).
	Keyword arguments:
	pixels -- An array of RGB pixels in the form (sample, RGB components).
	'''
	new_pixels = pixels.copy()
	R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]
	new_pixels[:, 0] = 0.299*R + 0.587*G + 0.114*B
	new_pixels[:, 1] = -0.169*R - 0.332*G + 0.500*B
	new_pixels[:, 2] = 0.500*R - 0.419*G - 0.081*B
	return new_pixels

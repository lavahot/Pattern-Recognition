import numpy as np

from bayes_classifier import  Distribution

def getMaximumLikelihood(training):
	''' Returns a Multivariate Gaussian of type bayes_classifier.Distribution that represents the data.
	TODO: Impliment the more generic version that does not assume independance across features.
	Keyword arguments:
	training -- An numpy array of features. It should be in the form (sample, feature)
	'''
	training = np.array(training)
	if len(training.shape) != 2:
		raise Exception('Training data must be an array of vectors.')
	d = training.shape[1]
	mu = []
	for i in range(d):
		mu.append(training[:, i].mean())
	covar = np.dot(training.transpose(), training)
	covar /= float(d)
	return Distribution(mu=mu, covar=covar)

import numpy as np
from numpy import linalg
import math

# Distribution
class Distribution:
	def __init__(self, mu, covar, prior=1.0):
		self.mu = np.array(mu, dtype='float')
		self.covar = np.array(covar, dtype='float')
		self.d = len(mu)
		self.prior = prior
		#  Dimension checking
		if len(self.mu.shape) != 1:
			raise Exception('mu must be a vector.')
		if len(self.covar.shape) != 2:
			raise Exception('covar must be a 2D matrix.')
		if self.covar.shape[0] != self.covar.shape[1]:
			raise Exception('covar must be a square matrix.')
		if self.d != self.covar.shape[0]:
			raise Exception('covar must have the same number of dimensions as the mu vector.')
	def getLogProb(self, x):
		if len(x.shape) != 1:
			raise Exception('x must be a vector.')
		if x.shape[0] != self.d:
			raise Exception('x must have the same number of dimensions as the distribution.')
		x = np.array(x, dtype='float')
		p_exp = -0.5 * np.dot((x - self.mu), np.dot(linalg.inv(self.covar), (x - self.mu)))
		p_bas = -1.0 * math.log(((2.0 * math.pi) ** (self.d/2.0)) * (linalg.det(self.covar)** (self.d/2)))
		p_pri = math.log(self.prior)
		return p_exp + p_bas + p_pri
	def __str__(self):
		return 'mu={mu}, covar={covar}, prior={prior}'.format(mu=self.mu,
		                                                      covar=self.covar,
		                                                      prior=self.prior)

def getClass(x, Classes):
	best = None, float('-inf')
	for C in Classes:
		p = C.getLogProb(x)
		if p > best[1]:
			best = C, p
	return best[0]

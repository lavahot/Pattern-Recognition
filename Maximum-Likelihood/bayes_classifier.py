import numpy as np
from numpy import linalg
import copy, math
import sympy

# Distribution
class Distribution:
    def __init__(self, mu, sigm, prior = 1.0):
        self.mu = copy.deepcopy(mu)
        self.sigm = copy.deepcopy(sigm)
        self.prior = prior
        self.dims = len(mu)
        if len(mu) != len(sigm):
            raise Exception('mu and sigm must have the same number of dims.')
    def genSamples(self, n):
        samples = np.empty((n, self.dims))
        for i in range(n):
            for j in range(self.dims):
                samples[i, j] = random.gauss(self.mu[j], self.sigm[j])
        return samples
    def getLogProb(self, val):
        logProb = math.log(self.prior)
        for i in range(self.dims):
            logProb += -(val[i] - self.mu[i])**2 / (2 * self.sigm[i]**2) + math.log(1/(self.sigm[i]*math.sqrt(2*math.pi)))
        return logProb
    def getLogProbFormula(self):
        X = []
        formula = math.log(self.prior)
        for i in range(self.dims):
            X.append(sympy.Symbol('x_'+str(i)))
        for i in range(self.dims):
            x = X[i]
            formula = formula - ((x - self.mu[i])**2 / (2 * self.sigm[i]**2)) + math.log(1/(self.sigm[i]*math.sqrt(2*math.pi)))
        return formula
    def __str__(self):
        return 'mu={mu}, sigm={sigm}, prior={prior}'.format(mu=self.mu,
                                                            sigm=self.sigm,
                                                            prior=self.prior)

# Predict class
def predictClass(x, classes):
    bestClass = None
    bestLogProb = -float('inf')
    for C in classes:
        logProb = C.getLogProb(x)
        if logProb > bestLogProb:
            bestLogProb = logProb
            bestClass = C
    return bestClass


# Bounds
def getBounds(A, B, linspace):
    equation = A.getLogProbFormula() - B.getLogProbFormula()
    sols = sympy.solve(equation, 'x_1')
    lines = []
    for sol in sols:
        line = [[], []]
        lines.append(line)
        for xVal in linspace:
            yVal = sol.subs('x_0', xVal)
            if not 'I' in str(yVal):
                xVal = float(xVal)
                yVal = float(yVal)
                line[0].append(xVal)
                line[1].append(yVal)
            pass
    return lines


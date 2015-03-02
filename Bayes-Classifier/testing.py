import numpy as np
import sympy
from numpy import linalg
import copy, math
from matplotlib import pyplot


part = 1
subPart = 'a'

# Test Paramters
A_MU = [1.0, 1.0]
B_MU = [4.0, 4.0]

if part == 1:
	A_SAMPLES_FNAME = '1-1.txt'
	B_SAMPLES_FNAME = '1-2.txt'
	A_SIGM = [1.0, 1.0]
	B_SIGM = [math.sqrt(1.0), math.sqrt(1.0)]
elif part == 2:
	A_SAMPLES_FNAME = '2-1.txt'
	B_SAMPLES_FNAME = '2-2.txt'
	B_MU = [4.0, 4.0]
	B_SIGM = [math.sqrt(4.0), math.sqrt(16.0)]
else:
	raise Exception('Unknown part')


if subPart == 'a':
	A_PRIOR = 0.5
	B_PRIOR = 0.5
elif subPart == 'b':
	A_PRIOR = 0.3
	B_PRIOR = 0.7	
else:
	raise Exception('Unknown subpart')


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


# Testing Parameters
A = Distribution(mu=A_MU,
                 sigm=A_SIGM,
                 prior=A_PRIOR)
B = Distribution(mu=B_MU,
                 sigm=B_SIGM,
                 prior=B_PRIOR)
Classes = [A, B]


# Load Data
with open(A_SAMPLES_FNAME, 'rb') as fp:
    A_samples = np.loadtxt(fp, dtype='float')
with open(B_SAMPLES_FNAME, 'rb') as fp:
    B_samples = np.loadtxt(fp, dtype='float')
print('A:', A_samples.shape)
print('B:', B_samples.shape)
# Plot the data
pyplot.scatter(B_samples[:,0], B_samples[:, 1], c='b', label='B')
pyplot.scatter(A_samples[:,0], A_samples[:, 1], c='r', label='A')
# Other
pyplot.xlim([-4, 12])
pyplot.ylim([-12, 16])
pyplot.legend()
pyplot.title('Sample Data')
pyplot.show()


# Get Bounds
linspace = np.linspace(-4, 12, 1000)
bounds = getBounds(A, B, linspace)
print(A.getLogProbFormula())
print(B.getLogProbFormula())


# Show Bounds
pyplot.scatter(B_samples[:, 0], B_samples[:, 1], c='b', label='B')
pyplot.scatter(A_samples[:, 0], A_samples[:, 1], c='r', label='A')
for bound in bounds:
    pyplot.plot(bound[0], bound[1], c='g')
pyplot.xlim([-4, 12])
pyplot.ylim([-12, 16])
pyplot.title('Decision Boundary')
pyplot.legend()
pyplot.show()


# Test A
right = 0
wrong = 0
for a in A_samples:
    w = predictClass(a, Classes)
    if w == A:
        right += 1
    else:
        wrong += 1
print('Class {c}, Right:{r}, Wrong:{w}'.format(c='A', r=right, w=wrong))
# Test B
right = 0
wrong = 0
for b in B_samples:
    w = predictClass(b, Classes)
    if w == B:
        right += 1
    else:
        wrong += 1
print('Class {c}, Right:{r}, Wrong:{w}'.format(c='B', r=right, w=wrong))


mu1 = np.array([[A_MU[0]], [A_MU[1]]])
mu2 = np.array([[B_MU[0]], [B_MU[1]]])
sigm1 = np.diag(A_SIGM) * np.diag(A_SIGM)
sigm2 = np.diag(B_SIGM) * np.diag(B_SIGM)
# Numpy Reference
# Mult: np.dot(A, B)
# Add/Sub: A-B
# Tranpose: A.transpose()
# Det: linalg.det(A)
# Inv: linalg.inv(A)
# Log: math.log(x)


def eval(diffmu,sigma1,sigma2,beta):
    return math.e ** -(beta * (1-beta) / 2 * np.dot(np.dot((diffmu).transpose(),        linalg.inv((1-beta) * sigma1 + beta * sigma2)),(diffmu))[0][0]         + 0.5 * math.log(linalg.det((1-beta) * sigma1 + beta * sigma2)         / (linalg.det(sigma1)**(1-beta) * linalg.det(sigma2)**beta)))

def ChernoffB(mu1,mu2,sigma1,sigma2, epsilon):
    diffmu=mu1-mu2
    curp=0.1
    ssize=0.00001
    accel=1.00001
    candidate = [0.0,0.0,0.0,0.0,0.0]
    candidate[0]=-accel
    candidate[1]=-1.0/accel
    candidate[2]=0.0
    candidate[3]=1.0/accel
    candidate[4]=accel
    while True:
        before = eval(diffmu, sigma1, sigma2, curp)
        best = -1
        bestscore=2
        for j in range(0,5):
            curp = curp+ssize*candidate[j]
            if curp < 0:
                curp=0
            elif curp >1:
                curp=1
            temp = eval(diffmu, sigma1, sigma2, curp)
            curp = curp-ssize*candidate[j]
            if(temp < bestscore):
                bestscore=temp
                best=j
        if candidate[best] != 0:
            curp=curp+ssize*candidate[best]
            ssize=ssize*candidate[best]
        curval=eval(diffmu, sigma1, sigma2, curp)    
        if abs(float(curval)-float(before)) < float(epsilon):
            return curp
    
chBound = ChernoffB(mu1,mu2,sigm1,sigm2, 1E-20)
bBound = 0.5


def f(B):
    global mu1, mu2, sigm1, sigm2
    return eval(mu1-mu2, sigm1, sigm2, B)
x = np.linspace(0, 1, 100)
y = list(map(f, x))
pyplot.plot(x, y)
pyplot.plot([0, chBound, chBound],
            [f(chBound), f(chBound), 0],
            label='Chernoff Bound')
pyplot.plot([0, bBound, bBound],
            [f(bBound), f(bBound), 0],
            label='Bhattacharya Bound')
pyplot.legend()
pyplot.title('Bounds')
pyplot.xlabel('$\\beta$')
pyplot.ylabel('$e^{-k(\\beta)}$')
pyplot.show()


print('Chernoff Bound:', chBound, f(chBound))
print('Bhattacharya Bound:', bBound, f(bBound))
print('Diff:', f(bBound)-f(chBound))

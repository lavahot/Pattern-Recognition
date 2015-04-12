from pca import *
from image_manipulation import *
import pickle, glob
import numpy as np
import matplotlib.pyplot as plt

MAX_RANK = 50

# Load pca
pca = pickle.load(open('training/pca.pkl', 'rb'))

# Load database
databaseProjections = np.loadtxt('training/projection.txt')
databaseIds = np.loadtxt('training/ids.txt').astype('int')
nDatabase = databaseIds.shape[0]
print('nDatabase:', nDatabase)

# Get testing files
testingFiles = glob.glob('fb_H/*.pgm')

# Get dimension info
h, w = readImage(testingFiles[0]).shape
d = h * w
nTesting = len(testingFiles)

# Load testing images
print('Loading testing images.')
testingVects = np.empty([d, nTesting])
for i in range(nTesting):
	img = readImage(testingFiles[i])
	x = vectorizeImage(img)
	testingVects[:, i] = x

# Get the ids
def getId(fname):
	for delim in ['/', '\\']:
		fname = fname.rsplit(delim, 2)[-1]
	return int(fname.split('_', 2)[0])
testingIds = list(map(getId, testingFiles))
testingIds = np.array(testingIds, dtype='int')

# Apply PCA
print('Applying PCA transformation.')
k = pca.k
testingProjections = np.empty([k, nTesting])
for i in range(nTesting):
	x = testingVects[:, i].copy()
	y = pca.project(x)
	testingProjections[:, i] = y

# Check proper reconstruction
x = testingVects[:, 0].copy()
x = pca.getReconstruction(x)
plt.imshow(img.reshape([h, w]), cmap='gray')
plt.show()

# Save testing info
np.savetxt('testing/projections.txt', testingProjections)
np.savetxt('testing/ids.txt', testingIds)	

# Get distance pairs
testingDists = [] # [nTesting, nDatabase, 2]
for i in range(nTesting):
	dists = []
	a = testingProjections[:, i].copy()
	for j in range(nDatabase):
		b = databaseProjections[:, j].copy()
		dist = pca.getMahalanobisDist(a, b)
		dists.append((dist, databaseIds[j]))
	dists.sort()
	testingDists.append(dists)

# Evaluate performance
matched = np.ones([nTesting, nDatabase])
for i in range(nTesting):
	selfId = testingIds[i]
	for j in range(nDatabase):
		otherId = testingDists[i][j][1]
		if selfId == otherId: # Matched
			break
		else: # Not matched, reduce the score
			matched[i, j] = 0

# Get performance at each rank
perf = np.empty(MAX_RANK)
for i in range(MAX_RANK):
	perf[i] = matched[:, i].sum() / nTesting

plt.xlabel('Performance')
plt.ylabel('Rank')
plt.title('Algorithm Performance')
plt.xlim([1.0, MAX_RANK])
plt.ylim([0.0, 1.0])
plt.plot(range(1, MAX_RANK+1), perf)
plt.show()
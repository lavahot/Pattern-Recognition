import glob, PIL.Image
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Max Rank
MAX_RANK = 50
W, H = 16, 20

# Images
databaseFiles = glob.glob('fa_L/*.pgm')
queryFiles = glob.glob('fb_L/*.pgm')
nDatabase = len(databaseFiles)
nQuery = len(queryFiles)
if MAX_RANK > nDatabase:
	MAX_RANK = nDatabase
# Map functions
def getId(fname):
	for delim in ['/', '\\']:
		fname = fname.rsplit(delim, 2)[-1]
	return int(fname.split('_', 2)[0])

def getVect(fname):
	global W, H
	img = PIL.Image.open(fname)
#	img = img.resize((W, H))
	img = np.array(img, dtype='float').reshape(W * H)
	return img

# Load images and ids
databaseIds = list(map(getId, databaseFiles))
queryIds = list(map(getId, queryFiles))


databaseVects = list(map(getVect, databaseFiles))
queryVects = list(map(getVect, queryFiles))

# Show a face
plt.title('Reduced Image, id={i}'.format(i=databaseIds[0]))
plt.imshow(databaseVects[0].reshape([H, W]).astype('uint8'), cmap='gray', clim=[0, 255]).set_interpolation('nearest')
plt.show()

# Get distance pairs
queryDists = []
for i in range(nQuery):
	dists = []
	for j in range(nDatabase):
			dist = la.norm(queryVects[i] - databaseVects[j])
			dists.append((dist, databaseIds[j]))
	dists.sort()
	queryDists.append(dists)
for i in range(10):
		plt.plot(np.array(queryDists[i])[:, 0])
plt.xlabel('Candidate #')
plt.ylabel('Distance')
plt.show()

# Evaluate performance
matched = np.ones([nQuery, nDatabase])
for i in range(nQuery):
	selfId = queryIds[i]
	for j in range(nDatabase):
		otherId = queryDists[i][j][1]
		if selfId == otherId: # Matched
			break
		else: # Not yet matched
			matched[i, j] = 0

#  Get performance at each rank
perf = np.empty(MAX_RANK)
for i in range(MAX_RANK):
	perf[i] = matched[:, i].sum() / nQuery

plt.xlabel('Performance')
plt.ylabel('Rank')
plt.title('Algorithm Performance')
plt.xlim([1.0, MAX_RANK])
plt.ylim([0.0, 1.0])
plt.plot(range(1, MAX_RANK+1), perf)
print(perf)
plt.show()
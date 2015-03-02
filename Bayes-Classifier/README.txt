Generating Data:
	python outputFname nSamples nDims mu1 mu2 ... mu_d sigm1 sigm2 ... sigmd
Example:
	python generate.py out.txt 128 2 1 2 2 2

Select Test:
	1. Open testing.py with a text editor
	2. Change line 8, to "part = x" where x is 1 or 2.
	3. Change line 9, to "subPart = x" where x is 'a' or 'b'.

Testing:
	python testing.py

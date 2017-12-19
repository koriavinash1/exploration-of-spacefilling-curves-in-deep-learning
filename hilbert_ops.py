import platform
import time
import numpy as np

def d2xy ( m, d ):

	## D2XY converts a 1D Hilbert coordinate to a 2D Cartesian coordinate.
	#  Parameters:
	#
	#    Input, integer M, the index of the Hilbert curve.
	#    The number of cells is N=2^M.
	#    Number of cells(pixels) = image dimensions
	#    0 < M.
	#    Input, integer D, the Hilbert coordinate of the cell.
	#    0 <= D < N * N.
	#    Output, integer X, Y, the Cartesian coordinates of the cell.
	#    0 <= X, Y < N.

	n = 2**m
	assert d <= n**2 - 1 
	x = y = 0
	t = d
	s = 1

	while ( s < n ):
		rx = 1 & (t / 2)
		ry = 1 & (t ^ rx)
		x, y = rot(s, x, y, rx, ry)
		x += s * rx
		y += s * ry
		t /= 4
		s *= 2

	return x, y

def rot ( n, x, y, rx, ry ):
	#  Parameters:
	#    Input, integer N, the length of a side of the square.  
	#    N must be a power of 2.
	#    Input/output, integer X, Y, the coordinates of a point.

	if ( ry == 0 ):
		#  Reflect.
		if ( rx == 1 ):
			x = n - 1 - x
			y = n - 1 - y
		#  Flip.
		t = x
		x = y
		y = t

	return x, y

def xy2d ( m, x, y ):

	#  Parameters:
	#    Input, integer M, the index of the Hilbert curve.
	#    The number of cells is N=2^M.
	#    0 < M.
	#    Output, integer D, the Hilbert coordinate of the cell.
	#    0 <= D < N * N.

	d = 0
	n = 2 ** m

	s = ( n // 2 )
	while s > 0:
		rx = (x & s) > 0
		ry = (y & s) > 0
		d += s * s * ((3 * rx) ^ ry)
		(x, y) = rot(s, x, y, rx, ry)
		s /= 2
	return d

def find_hilbert_index(imagedim):
	# helps in finding hilbert index
	# Input image dimension
	# Output int hilbert
	idx = 0
	for i in range(imagedim):
		if 2**i > imagedim:
			idx =  0
		elif 2**i == imagedim:
			# print "Hilbert Index of given image: {}".format(i)
			idx = i
			break
	if not idx:
		print "Improper Image dimension: {}, no power found".format(imagedim)
		return ValueError
	return idx

def image2signal(image):
	#  Parameters:
	#    Input, numpy array normal image 
	#    Output,  numpy array hilbert image
	#    Output image dimension normal image**2

	assert image.shape[0] == image.shape[1]
	try:
		channels = image.shape[2]
	except:
		channels = 1

	hilbert_index = find_hilbert_index(image.shape[0])
	hilbert_image = [np.ones(image.shape[0]*image.shape[1]) for i in range(channels)]

	image = np.transpose(image).reshape(channels, image.shape[0], image.shape[0])
	for y in range(image.shape[2]-1, -1, -1):
		for x in range(0, image.shape[1]):
			d = xy2d(hilbert_index, x, y)
			for i in range(channels):
				hilbert_image[i][d] = image[i][x][y]
	return np.array(hilbert_image).T

def signal2image(signal):
	#TODO: fix bug in multichannel input

	#  Parameters:
	#    Input, numpy array hilbert image 
	#    Output,  numpy array normal image
	#    Output image dimension sqrt(hilbert image)
	try:
		channels = signal.shape[1]
	except:
		channels = 1
	img_dim = int(np.sqrt(signal.shape[0]))
	hidx = find_hilbert_index(img_dim)
	orig_image = np.ones((channels, img_dim, img_dim), dtype="float32")
	signal = signal.reshape(channels, img_dim*img_dim)

	for d in range(0, img_dim**2):
		x, y=d2xy(hidx, d)
		for i in range(channels):
			orig_image[i][x][y] = signal[i][d]

	return orig_image.T
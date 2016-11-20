import numpy as np
import csv
import argparse
import cv2


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--indexfile", required = True,
	help = "path of features csv file")
parser.add_argument("-q", "--queryimage", required = True,
	help = "Path to the query image")
parser.add_argument("-p", "--photosfolder", required = True,
	help = "folder with all photos")
parser.add_argument("-t", "--Tops", required = True,
	help = "# of display")
args = vars(parser.parse_args())

class featuregetter:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins

	def describe(self, image):
		# convert the image to the HSV color space and initial feature
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		# extract a color histogram from the elliptical region and
		# update the feature vector
		hist = self.histogram(image)
		features.extend(hist)
		# return the feature vector
		return features

	def histogram(self, image):
		# extract features 3D/whole image/bin numbers A x B x C/hsv 0-180 
		# 0-256 0-256
		hist = cv2.calcHist([image], [0, 1, 2], None, self.bins,
			[0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist).flatten()

		# return the histogram
		return hist

class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath

	def search(self, queryFeatures, limit =int(args["Tops"])):
		# build a dictionary for result
		fileChi2_Dict = {}

		# open file and read
		with open(self.indexPath) as f:
			# CSV reader
			reader = csv.reader(f)

			# loop for each row
			for row in reader:
				# build a features list with each component in each row of 
				# feature csv file
				features = [float(x) for x in row[1:]]

				# calculate chi square distance
				Chi2d = self.chi_square_distance(features, queryFeatures)

				# set dict: photoID as the key d as value
				fileChi2_Dict[row[0]] = Chi2d

			# close file
			f.close()

		# exchange key value,then sort the dict with value in front
		rankedresults = sorted([(a, b) for b, a in fileChi2_Dict.items()])

		# return useful results
		return rankedresults[:limit]
	
	# Chi Square Distance function
	def chi_square_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d



# initialize the featuregetter
# fg is a object of featuregetter class with hue bin 3, saturation bin 12 and value bin 3
fg = featuregetter((8, 12, 3))

# read query image 
queryimage = cv2.imread(args["queryimage"])

#extract features
queryfeatures = fg.describe(queryimage)

# perform the search
#searcher is a object of Searcher class, argument is the csv file
searcher = Searcher(args["indexfile"])

#argument is the query's features and return 5 similar images in result list
results = searcher.search(queryfeatures)

# display the queryimage, argument is pass through
cv2.imshow("Query image", queryimage)

# loop over the results in results list(5items)
for (Chi2_ditance, photoname) in results:
	# read the image in results by looking the path
	displayresult = cv2.imread(args["photosfolder"] + "/" + photoname)
	# Display
	cv2.imshow("Result images", displayresult)
	cv2.waitKey(0)

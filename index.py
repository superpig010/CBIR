from featuregetter import featuregetter
import argparse
import glob
import cv2

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
parser.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(parser.parse_args())

# initialize the color descriptor
fg = featuregetter((8, 12, 3)) #HSV

# open the output index file for writing
output = open(args["index"], "w")  #index is the file name

# use glob to grab the image paths and loop over them
for image in glob.glob(args["dataset"] + "/*.png"): #any file in path with g.g(x.png)
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = image[image.rfind("/") + 1:] #get all letters after "/", image is a path string
	image = cv2.imread(image)   #read image by opencv funtion

	# describe the image
	features = fg.describe(image)  #using the ColorDescriptor to extract features

	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))

# close the index file
output.close()

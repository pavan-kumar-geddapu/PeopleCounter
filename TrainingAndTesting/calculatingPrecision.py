from numpy import *

outputFile = open("/home/pavan/projects/PeopleCounter/TrainingAndTesting/testImageOutput.txt","r")
imageOutputList = outputFile.readlines()
outputFile.close()

totalImagesCount = 0
currentPrecision = double(0.0)
for i in imageOutputList:
	totalImagesCount += 1
	arr = i.split()
	if int(arr[2]) != 0:
		currentPrecision += double( 100 - ( ( 1-abs ( int(arr[2])- int(arr[1]) ) )/ int(arr[2]) ) * 100 )
	else:
		currentPrecision += 100

avgPrecision = double(currentPrecision/(totalImagesCount * 100)) * 100

print("average precision is : {}".format(avgPrecision))

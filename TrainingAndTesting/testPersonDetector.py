import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

classesFile = "/home/pavan/projects/PeopleCounter/TrainingAndTesting/coco.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "/home/pavan/projects/PeopleCounter/TrainingAndTesting/yolov3.cfg";
#modelWeights = "/home/pavan/projects/PeopleCounter/TrainingAndTesting/weights/darknet-yolov3_400.weights";
modelWeights = "/home/pavan/projects/PeopleCounter/TrainingAndTesting/weights/yolov3.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

outputFile = open("testImageOutput.txt","w+") #to write actual and predicted count of each testing image

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, imageName, objectCount):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            if confidence > confThreshold and classId == 0:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count = 0
    for i in indices:	
        count += 1
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    outputFile.write("{} {} {}".format(imageName, count, objectCount))
    outputFile.write("\n")

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

testDataFile = open("/home/pavan/projects/PeopleCounter/DataSet/person_test.txt","r")
testFileList = testDataFile.readlines()
testDataFile.close()

imgCount = 0
for image in testFileList:
	print(imgCount)
	image = image[:-1]
	
	cap = cv.VideoCapture(image)
	hasFrame, frame = cap.read()
	blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
	net.setInput(blob)
	outs = net.forward(getOutputsNames(net))
	imageName = image[-20:]

	labelFileName = '/home/pavan/projects/PeopleCounter/DataSet/labels/'+imageName[:-4]+'.txt'
	labelFile = open(labelFileName,"r")
	fileLines = labelFile.readlines()
	objectCount = 0
	for i in fileLines:
		objectCount += 1
	labelFile.close()

	outputFileName = "/home/pavan/projects/PeopleCounter/TrainingAndTesting/output/"+imageName
	postprocess(frame, outs, imageName, objectCount)
	cv.imwrite(outputFileName, frame.astype(np.uint8))
	imgCount += 1
	
outputFile.close()


import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import pprint

from sort import *


ROI = [(100,100),(1880,100),(100,980),(1880,980)]

def isInside(objCenter):

    rc = ROI.copy()
    x = objCenter[0]
    y = objCenter[1]

    x1 = (objCenter[1]-rc[0][1])*((rc[2][0]-rc[0][0])/(rc[2][1]-rc[0][1])) + rc[0][0]
    x2 = (objCenter[1]-rc[1][1])*((rc[3][0]-rc[1][0])/(rc[3][1]-rc[1][1])) + rc[1][0]
    y1 = (objCenter[0]-rc[0][0])*((rc[1][1]-rc[0][1])/(rc[1][0]-rc[0][0])) + rc[0][1]
    y2 = (objCenter[0]-rc[2][0])*((rc[3][1]-rc[2][1])/(rc[3][0]-rc[2][0])) + rc[2][1]

    if x>x1 and x<x2 and y>y1 and y<y2 :
        return True
    return False


tracker = Sort()
memory = {}

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


#labelsPath = '/home/pavan/projects/PeopleCounter/TrainingAndTesting/classes.names'
#config_file_abs_path = "/home/pavan/projects/PeopleCounter/TrainingAndTesting/darknet-yolov3.cfg"
#weights_file_abs_path = "/home/pavan/projects/PeopleCounter/TrainingAndTesting/weights/darknet-yolov3_400.weights"

labelsPath = '/home/pavan/projects/PeopleCounter/VideoImplementation/yolo-coco/coco.names'
config_file_abs_path = "/home/pavan/projects/PeopleCounter/VideoImplementation/yolo-coco/yolov3.cfg"
weights_file_abs_path = "/home/pavan/projects/PeopleCounter/VideoImplementation/yolo-coco/yolov3.weights"

LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_file_abs_path, weights_file_abs_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(ln)

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

outputPath = args["output"]
outputPath = outputPath[:-16]
print(outputPath)

files = glob.glob("{}*.png".format(outputPath))
for f in files:
    os.remove(f)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

frameIndex = 0

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    print(frameIndex)

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        print("height, width : {}, {}".format(H,W))

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])


    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])


    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)

    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    counter = 0

    if len(boxes) > 0:
        i = int(0)
        for indexID in memory :
            box = memory[indexID]
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            objCenter = (int(x + (w-x)/2), int(y + (h-y)/2))
            if isInside(objCenter):
                counter += 1

            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1



    # draw ROI
    cv2.rectangle(frame, ROI[0], ROI[3], (255,255,0), 2)


    # draw counter
    cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)

    # saves image file
    cv2.imwrite("{}frame-{}.png".format(outputPath,frameIndex), frame)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * 300))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

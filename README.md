# PEOPLE COUNTER

Detecting and counting people from a surveillance video footage using YOLOv3 neural net framework.

Inspired by : https://www.learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/

## Dataset Building
Get the relevant OpenImages files needed to locate images of our interest
```
wget https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv
```
Download the images from OpenImagesV4 . This will download the images into **JPEGImages** folder and corresponding label files into **lables** folder. Here you can use your own data by just keeping relevant files in those two folders.
```
python3 getDataFromOpenImages_snowman.py
```
splitting training and testing data. This will split image paths into two text files **person_train.txt** and ** person_test.txt**
```
python3 splitTrainAndTest.py /home/pavan/projects/PeopleCounter/DataSet/JPEGImages
```

## Installing Darknet
```
cd ~
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
Get pretrained model
```
wget https://pjreddie.com/media/files/darknet53.conv.74 -O ~/darknet/darknet53.conv.74
```

## Training The Model
fill correct paths in **darknet.data** 
```
cd ~/darknet
./darknet detector train /home/pavan/projects/PeopleCounter/TrainingAndTesting/darknet.data /home/pavan/projects/PeopleCounter/TrainingAndTesting/darknet-yolov3.cfg ./darknet53.conv.74 > /home/pavan/projects/PeopleCounter/TrainingAndTesting/train.log
```
you can check the training log using this command 
```
grep "avg" /home/pavan/projects/PeopleCounter/TrainingAndTesting/train.log
```
you can plot loss curve using **train.log** data.
```
python3 plotTrainLoss.py /home/pavan/projects/PeopleCounter/TrainingAndTesting/train.log
```
final weights will be stored in weights folder. You have to create a weights folder before implementing above steps.

## Testing The Model
```
python3 testPersonDetector.py
```
relevent results will be stored in **testImageOutput.txt** which can be used to calculate precision
```
python3 calculatingPrecision.py
```

## Testing The Model On Video
Implementation for Line of interest algorithm
```
python3 implementationLOI.py --input input/test_video_1.mp4 --output output/test_video_1/LOI/test_video_1.avi
```
Implementation for Region of interest algorithm
```
python3 implementationROI.py --input input/test_video_1.mp4 --output output/test_video_1/ROI/test_video_1.avi
```
You can download coco dataset (which includes trained weights, configuration file, 80 different classes) to compare your version with it.

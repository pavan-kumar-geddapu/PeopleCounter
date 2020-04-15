			PEOPLE COUNTER

Get the relevant OpenImages files needed to locate images of our interest

wget https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv

wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv


Download the images from OpenImagesV4

python3 getDataFromOpenImages_snowman.py


SPLITTING TRAIN AND TEST DATA SETS :

python3 splitTrainAndTest.py /home/pavan/projects/PeopleCounter/DataSet/JPEGImages


INSTALLING DARKNET AND COMPILING :

cd ~
git clone https://github.com/pjreddie/darknet
cd darknet
make


GET PRETRAINED MODEL :

wget https://pjreddie.com/media/files/darknet53.conv.74 -O ~/darknet/darknet53.conv.74


TRAINING THE MODEL :

fill correct paths in darknet.data

cd ~/darknet

./darknet detector train /home/pavan/projects/PeopleCounter/TrainingAndTesting/darknet.data /home/pavan/projects/PeopleCounter/TrainingAndTesting/darknet-yolov3.cfg ./darknet53.conv.74 > /home/pavan/projects/PeopleCounter/TrainingAndTesting/train.log

grep "avg" /home/pavan/projects/PeopleCounter/TrainingAndTesting/train.log

python3 plotTrainLoss.py /home/pavan/projects/PeopleCounter/TrainingAndTesting/train.log


TESTING THE MODEL :

python3 testPersonDetector.py

python3 calculatingPrecision.py


TESTING ON VIDEO :

python3 implementationLOI.py --input input/test_video_1.mp4 --output output/test_video_1/LOI/test_video_1.avi

python3 implementationROI.py --input input/test_video_1.mp4 --output output/test_video_1/ROI/test_video_1.avi


# Face Recognition using Dlib and Tensorflow


## The program is divided into 6 modules:

### 1, FiletoNumpy.py 
		The database initially had labels only for the detected regions of faces which is not enough for the task of recognition.
		Therefore the database has to be labelled manually in a text file before proceeding with face recognition. 
		The following module obtains a numpy array from the text file annotation of the database.
		
### 2, facemodel.py
		Loads resnet model, face descriptor and detector from the data folder and return face embedding and num of faces 
		detected.The python file is used as a helper.
		
### 3, knnModel.py
		Contains code for a knn model written in Tensorflow. It can function with query descriptor or can operate with 
		test,train data during training time.
		
### 4, main.py
		Manages all the tasks of training and checking accuracy for the test/train split. The program has to be run
		whenever a new person has been added into the database.The vectors and labels of new person are stored data folder
		
### 5, util.py
		Helper module that contains most of I/O tasks. This helps keep the code clean and relevant.
		
### 6, classifier.py
		Should be run in production mode in a loop. The module can detect faces from given image (provided person is in database) 
		and manages scenarios of multiple face detections as well.
		
# Steps for adding new person to Database: 

#### Add 5+ images of person with a front pose in the database [ Dataset/]
#### Add the name of the image file (.jpg) into faceLabels.txt along with a label number [data/]
		
		For example:  If the person's images are sanjeev1.jpg,sanjeev2.jpg...
						Add the following lines into faceLabels.txt: 
									............. , .....
									sanjeev1.jpg, 25
									sanjeev2.jpg, 25 
						Here sanjeevN.jpg is the name of the image which will be read
								25 is the id associated with the person
								
#### Delete (or backup) Ximages.npy, Ylabels.npy, knnData.npy from the previous database from [data/]
#### Run FileToNumpy.py
#### Run main.py
		This will the updated database : Ximages.npy, Ylabels.npy, knnData.npy in [data/]
		
##### _For inference: Run classifier.py , change path of test to any of the images in the database_

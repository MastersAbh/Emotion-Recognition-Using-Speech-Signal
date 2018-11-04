import librosa
import os
import numpy
import scipy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
numpy.seterr(all='warn')

def getSegment(frames, start, end):
	rows = frames[start : end+1, ]
	return numpy.hstack(rows)
	
	
audios = []
emotions = []
audios2 = []
emotions2 = []
emo = []
emo2 = []
emotion_class_map = {'e' : 0, 'p' : 1, 's' : 2, 'a' : 3}
em = {0 : "anger", 1 : "happy", 2 : "sadness", 3 : "fear"}
num_emotions = len(emotion_class_map)

folder_path = "F:/train/"
folder_path2 = "F:/test/"
print("Reading train data")
for filename in os.listdir(folder_path):
    print(filename)
    filepath = folder_path + filename
    y, sr =librosa.load(filepath)
    emotion = filepath[-9] #2nd last character of file exluding extension name wav
    emotion_class = emotion_class_map[emotion]
    emotions.append(emotion_class)
    audios.append(y)

print("Reading test data")
for filename in os.listdir(folder_path2):
    print(filename)
    filepath = folder_path2 + filename
    y, sr =librosa.load(filepath)
    emotion = filepath[-9] #2nd last character of file exluding extension name wav
    emotion_class = emotion_class_map[emotion]
    emotions2.append(emotion_class)
    audios2.append(y)

sample_rate = 16000 # in hertz
frameDuration = 0.025 #duration in seconds
hopDuration = 0.010
frameSize = int(sample_rate*frameDuration)
hopSize = int(sample_rate*hopDuration)
featuresPerFrame = 13
framesPerSegment = 25
featuresPerSegment = featuresPerFrame * framesPerSegment
segmentHop = 13

X = numpy.empty((0, featuresPerSegment))
Y = numpy.empty((0, num_emotions))
X2 = numpy.empty((0, featuresPerSegment))
Y2 = numpy.empty((0, num_emotions))
print("Processing train dataset")
for i in range(len(audios)):
	print(str(em[emotions[i]]) + " " + str(i))
	audio = audios[i]
	output = emotions[i]
	output_vec = numpy.zeros((1, num_emotions))
	output_vec[0][output] = 1
	frames = numpy.empty((featuresPerFrame, )) #each index stores a list of 53 features for that frame
	start = 0
	countf =0 
	while True:
		end = start + frameSize
		countf = countf+1
		frame = numpy.zeros((frameSize,))
		if end >= len(audio) :
			for j in range(start,len(audio)):
				frame[j-start]=audio[j]
			for j in range(len(audio),end):
				frame[j-start]=0.0
			mfcc_coeffs = librosa.feature.mfcc(y=frame,sr=sample_rate,n_mfcc=13)
			frame_features = numpy.append(mfcc_coeffs.mean(axis=1), [])
			if numpy.isnan(mfcc_coeffs).any() :
				print("nan\nnan\n")
				exit()
			frames = numpy.vstack([frames,frame_features])
			break
		
		#numpy.append(frame,99)
		for j in range(start,end):
			frame[j-start]=audio[j]
		#print(frame)
		start = start + hopSize
		mfcc_coeffs = librosa.feature.mfcc(y=frame,sr=sample_rate,n_mfcc=13)
		frame_features = numpy.append(mfcc_coeffs.mean(axis=1), [])
		if numpy.isnan(mfcc_coeffs).any() :
			print("nan\nnan\n")
			exit()
		frames = numpy.vstack([frames,frame_features])
	start_segment = 0
	#print(countf)
	count = 0
	while True:
		end_segment = start_segment + framesPerSegment - 1 #center = 13, left=1-12, right=14-25
		#print(start_segment)
		if end_segment >= len(frames) :
			break
		count = count +1
		segment = getSegment(frames, start_segment, end_segment)
		start_segment = start_segment + segmentHop -1 #segmentSize = 13
		X = numpy.vstack([X, segment])
		emo = numpy.append(emo, em[output])
		Y = numpy.vstack([Y, output_vec])
	#print(count)
X_scaled = preprocessing.scale(X)
scipy.io.savemat('X.mat', {'X' : X})
scipy.io.savemat('Y.mat', {'Y' : Y})
scipy.io.savemat('X_scaled.mat', {'X_scaled' : X_scaled})

data = scipy.io.loadmat("X_scaled.mat")

for i in data:
    if '__' not in i and 'readme' not in i:
        numpy.savetxt(("file.csv"),data[i],delimiter=',')
        
        
print("Processing test dataset")

for i in range(len(audios2)):
	print(str(em[emotions2[i]]) + " " + str(i))
	audio = audios2[i]
	output = emotions2[i]
	output_vec = numpy.zeros((1, num_emotions))
	output_vec[0][output] = 1
	frames = numpy.empty((featuresPerFrame, )) #each index stores a list of 53 features for that frame
	start = 0
	while True:
		end = start + frameSize
		frame = numpy.zeros((frameSize,))
		if end >= len(audio) :
			for j in range(start,len(audio)):
				frame[j-start]=audio[j]
			for j in range(len(audio),end):
				frame[j-start]=0
			mfcc_coeffs = librosa.feature.mfcc(y=frame,sr=sample_rate,n_mfcc=13)
			frame_features = numpy.append(mfcc_coeffs.mean(axis=1), [])
			if numpy.isnan(mfcc_coeffs).any() :
				print("nan\nnan\n")
				exit()
			frames = numpy.vstack([frames,frame_features])
			break
		frame = numpy.zeros((frameSize,))
		for j in range(start,end):
			frame[j-start]=audio[j]
            
		start = start + hopSize
		mfcc_coeffs = librosa.feature.mfcc(y=frame,sr=sample_rate,n_mfcc=13)
		frame_features = numpy.append(mfcc_coeffs.mean(axis=1), [])
		if numpy.isnan(mfcc_coeffs).any() :
			print("nan\nnan\n")
			exit()
		frames = numpy.vstack([frames,frame_features])
	start_segment = 0
	while True:
		end_segment = start_segment + framesPerSegment - 1 #center = 13, left=1-12, right=14-25
		if end_segment >= len(frames) :
			break
		segment = getSegment(frames, start_segment, end_segment)
		start_segment = start_segment + segmentHop -1 #segmentSize = 13
		X2 = numpy.vstack([X2, segment])
		emo2 = numpy.append(emo2, em[output])
		Y2 = numpy.vstack([Y2, output_vec])
		
X_scaled2 = preprocessing.scale(X2)
scipy.io.savemat('X2.mat', {'X2' : X2})
scipy.io.savemat('Y2.mat', {'Y2' : Y2})
scipy.io.savemat('X_scaled2.mat', {'X_scaled2' : X_scaled2})

data = scipy.io.loadmat("X_scaled2.mat")
for i in data:
    if '__' not in i and 'readme' not in i:
        numpy.savetxt(("file2.csv"),data[i],delimiter=',')


print("Building NN")

seed = 7
numpy.random.seed(seed)   #run the same code again and again and get the same result.
    
def label_encoder(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y
 
def create_model():
    model = Sequential()
    model.add(Dense(15, input_dim=325, activation="relu", kernel_initializer="normal"))
    model.add(Dense(15, activation="relu", kernel_initializer="normal"))
    model.add(Dense(15, activation="relu", kernel_initializer="normal"))
    model.add(Dense(4, activation="softmax", kernel_initializer="normal"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model



filename = 'file.csv'
filename2 = 'file2.csv'
print("reached")
dataframe = pandas.read_csv(filename, header = None)

dataframe[len(dataframe.columns)]=emo
dataset = dataframe.values

dataframe2 = pandas.read_csv(filename2, header = None)

dataframe2[len(dataframe2.columns)]=emo2
dataset2 = dataframe2.values

n_inputs = len(dataset[0]) -1
n_outputs = len(set([row[-1] for row in dataset]))

n_inputs2 = len(dataset2[0]) -1
n_outputs2 = len(set([row[-1] for row in dataset2]))
X = dataset[:,0:n_inputs].astype(float)
Y = dataset[:,n_inputs]
Y = numpy.append(Y,"happy")
Y = numpy.append(Y,"fear")
Y = numpy.append(Y,"anger")
Y = numpy.append(Y,"sadness")
X2 = dataset2[:,0:n_inputs2].astype(float)
Y2 = dataset2[:,n_inputs2]
Y2 = numpy.append(Y2,"happy")
Y2 = numpy.append(Y2,"fear")
Y2 = numpy.append(Y2,"anger")
Y2 = numpy.append(Y2,"sadness")
dummy_y = label_encoder(Y)
dummy_y2 = label_encoder(Y2)
l = len(dummy_y)-4
dummy_y = dummy_y[0:l]
l = len(dummy_y2)-4
dummy_y2 = dummy_y2[0:l]
model = create_model()
model.fit(X, dummy_y, batch_size = 5, epochs=20)
score, acc = model.evaluate(X2, dummy_y2,batch_size=5)
print("Accuracy: "+str(acc*100))



	


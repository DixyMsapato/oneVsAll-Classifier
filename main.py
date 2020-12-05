import math
import numpy
import csv
from matplotlib import pyplot as plt
import os
import random
import scipy.optimize as op

#Assuming the data is all integer data
#And of a particular format, the getTrainingData needs modifications
#In particular, the target/class values are in the first column
#Depending on number of features of data
#We convert the data from getTrainingData from string format to

def getTrainingData():
    #call input function
    filepath = input('Please enter the file path for your training data: ')
    if os.path.exists(filepath):
        with open(filepath, 'r') as trainingData:
            csv_reader = csv.reader(trainingData, delimiter = ',')
            header = next(csv_reader)
            #load the data into a list of lists
            data = [row for row in csv_reader]
            colNum = len(data[0])

            #trueColNum ignores the columns with non-numerical data
            
            trueColNum = colNum

            intData = []

            for line in data:
                intLine = []
                counter = 0
                # while loop which converts line into a line of float instead of str values
                # we want float not into so that when we normalise, division by
                # standard deviation won't result in zero entries
                while counter < trueColNum:
                    intLine.append(float(line[counter]))
                    counter = counter + 1

                intData.append(intLine)
            # This turns 2d array into a numpy 2d array i.e matrix
            # Thus can now use numpy operations on our data!
            return numpy.array(intData)
    elif filepath == 'quit':
        return
    else:
        print("The path you entered does not exists, please try again or type quit to exit the program")
        #Call getTrainingData() again.
        getTrainingData()

trainingData = getTrainingData()

#This turns 2d array into a numpy 2d array i.e matrix
#Thus can now use numpy operations on our data!

#We first split the data into training variables X and output y
#the target of our MNIST training data is in the first column of the trainingData
yTarget = trainingData[:,[0]]
#create Theta_Zero column on int 1s
#We multiply by (0.99/255 +0.01) to adjust the ones to the scaling done to the mnist data
thetaZero = (0.01388235294)*numpy.ones((trainingData.shape[0],1),int)
#Remove the y values from training data
Xprime = numpy.delete(trainingData,0,1)
#The design matrix X
X = numpy.append(thetaZero, Xprime,1)

#display random ten digits from the data
print("Checking the data,by displaying 10 random training examples...")
image_width = int(math.sqrt(Xprime.shape[1]))
for i in range(0,10):
    j = random.randint(1, Xprime.shape[0])
    img = Xprime[j].reshape((image_width,image_width))
    plt.imshow(img, cmap="Greens")
    plt.show()
plt.close()
print("End of data check.")
#first we define the sigmoid and logistic function
def sigmoid(z):
    return 1/(1+numpy.exp(-z))


def logarithm(z):
    g = numpy.log(z)
    return g

def costFunction(theta,X,y,regCoeff):
    #m = number of training examples i.e length(y)
    #n = number of features
    n = numpy.shape(X)[1]
    theta = numpy.reshape(theta, (n, 1))
    m = numpy.shape(X)[0]
    y = numpy.reshape(y, (m, 1))
    #Let h denote the hypothethis
    h = sigmoid(X.dot(theta))
    truncTheta = theta[1:]
    regTerm  = (regCoeff/(2*m))*sum(truncTheta*truncTheta)
    J = (-1 / m) * (numpy.transpose(y).dot(logarithm(h))+(numpy.transpose((1-y))).dot(logarithm(1-h)))
    regJ =  J + regTerm
    return regJ

#gradient
#data: training data X is an mxn array, m =num training examples, n= num features
#y is an mx1 vector with the y values for the training examples
#theta is an nx1 vector
def gradient(theta,X,y,regCoeff):
    n = numpy.shape(X)[1]
    theta = numpy.reshape(theta,(n,1))
    m = numpy.shape(X)[0]
    y = numpy.reshape(y,(m,1))
    truncTheta = theta[1:]
    grad = (1/m)*numpy.transpose(X).dot((sigmoid(X.dot(theta))-y))
    thetazero = numpy.zeros((1,1))
    regTerm = numpy.append(thetazero,truncTheta,0)
    regGrad = grad + (regCoeff/m)*regTerm

    #need to unroll the grad vector to use the scipy mininimize methods
    return regGrad.ravel()


#Step 1 - shuffle the data the data, call it X
#X = numpy.random.permutation(X)
y = trainingData[:,[0]]
theta = numpy.ones((X.shape[1],1))

#Set the regularisation coefficient to 0.1
regCoeff = 0.01

print("initiating oneVsAll...")
print("Running oneVsAll......")
#input X,y
#num_Classes = number of classes to classify
#regularisation coefficient regCoeff
def oneVsAll(X,y,num_class):
    # m = number of training examples i.e length(y)
    # n = number of features
    n = numpy.shape(X)[1]
    m = numpy.shape(X)[0]
    #set shape of y to (m,1) instead of (m,)
    y = numpy.reshape(y, (m, 1))

    all_theta = numpy.zeros((num_class,n))

    for i in range(0,num_class):
        #vector of shape (m,1) like y
        a = i * numpy.ones((m,1))
        #boolean vector, creating 2 classes
        #class i vs All
        b = (y == a)
        #regression coefficient lambda
        regCoeff = 0.1
        initial_theta = numpy.zeros((n,1))
        Result = op.minimize(fun=costFunction,
                             x0=initial_theta,
                             args=(X, b , regCoeff),
                             method='TNC',
                             jac=gradient)
        optimal_theta = Result.x
        #print("This is Result.x,number",i)
        #print(Result.x)
        #print("Next Result.x")
        all_theta[i] = optimal_theta
    return all_theta

all_theta = oneVsAll(X,y,10)
print("oneVsAll complete!")

#PREDICT Predict whether the test_digits is 0,1,2,3,4,5,6,7,8,9 using learned logistic
#design matrix X
#regression parameters all_theta
def prediction(X,all_thetas):

    m = numpy.shape(X)[0]
    n = numpy.shape(X)[1]
    num_labels = numpy.shape(all_thetas)[0]

    predictor = numpy.zeros((m,1))
    thetasTranspose =  numpy.transpose(all_thetas)
    probabilityMatrix  = sigmoid(X.dot(thetasTranspose))

    indices = numpy.argmax(probabilityMatrix,axis=1)
    predictor = indices

    return predictor

#Get the testing data
def getTestingData():
    #call input function
    filepath = input('Please enter the file path for your testing data: ')
    if os.path.exists(filepath):
        with open(filepath, 'r') as testingData:
            csv_reader = csv.reader(testingData, delimiter = ',')
            header = next(csv_reader)
            #load the data into a list of lists
            data = [row for row in csv_reader]
            colNum = len(data[0])

            trueColNum = colNum

           #This for loop changes the data entries from string to int

            intData = []

            for line in data:
                intLine = []
                #print(type(line[0]))
                counter = 0
                # while loop which converts line into a line of float instead of str values
                # we want float not into so that when we normalise, division by
                # standard deviation won't result in zero entries
                while counter < trueColNum:
                    intLine.append(float(line[counter]))
                    counter = counter + 1

                intData.append(intLine)
            # This turns 2d array into a numpy 2d array i.e matrix
            # Thus can now use numpy operations on our data!
            return numpy.array(intData)
    elif filepath == 'quit':
        return
    else:
        print("The path you entered does not exists, please try again or type quit to exit the program")
        #Call getTestingData() again.
        getTestingData()

testingData = getTestingData()

#This turns 2d array into a numpy 2d array i.e matrix
#Thus can now use numpy operations on our data!

#We first split the data into test variables X and output y
#the target of our MNIST test data is in the first column of the testingData
yTarget = testingData[:,[0]]
#create Theta_Zero column on int 1s
thetaZero = (0.01388235294)*numpy.ones((testingData.shape[0],1),int)
#Remove the y values from training data
XprimeTest = numpy.delete(testingData,0,1)
#print(Xprime)
#The design matrix X
XTest = numpy.append(thetaZero, XprimeTest,1)

#display random ten digits from the data
image_width = int(math.sqrt(XprimeTest.shape[1]))
e = int(numpy.shape(XTest)[1])
test_digits = numpy.ones((10,e))
print("Displaying 10 randomly selected test examples...")
for i in range(0,10):
    j = random.randint(1, XprimeTest.shape[0])
    img = XprimeTest[j].reshape((image_width,image_width))
    #we will use test digits to test predictor
    #test_digits
    test_digits[i] = XTest[j]
    print(yTarget[j])
    plt.imshow(img, cmap="Greens")
    plt.show()
plt.close()
print("The predicted values are according to the oneVsAll algorithm are...")
print(prediction(test_digits,all_theta))




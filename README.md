# oneVsAll-Classifier
A python implementation of the "one vs all" multi-class classifier, using scipy.optmise 'TNC' algorithm for the logistic linear regression.

Input: 
      -Initial input. Comma-separated-values (csv) formatted training data. The code is written assuming the target variables are in row 0 of the data set  
The code checks the data and displays 10 random training examples  
The code then performs oneVsAll classification, outputting the optimised parameter for logistic regression.
      -Second input. The code then asks for Comma-separated-values (csv) formatted test data, it randomly selects 10 values to test. 
 The code displays the 10 randomly chosen test examples then performs classifiers the examples, and prints out a vector of the classes of the data.     
 
I used the mnist data sets for development and testing. Please find the mnist_train.csv data and mnist_test.csv data attached. 
About the data obtained from, https://www.python-course.eu/neural_network_mnist.php 
-The MNIST database (Modified National Institute of Standards and Technology database) of handwritten digits consists of a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. Additionally, the black and white images from NIST were size-normalized and centered to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.
This database is well liked for training and testing in the field of machine learning and image processing. It is a remixed subset of the original NIST datasets. One half of the 60,000 training images consist of images from NIST's testing dataset and the other half from Nist's training set. The 10,000 images from the testing set are similarly assembled.
The MNIST dataset is used by researchers to test and compare their research results with others. The lowest error rates in literature are as low as 0.21 percent.1

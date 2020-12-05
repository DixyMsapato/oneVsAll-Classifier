# oneVsAll-Classifier
A python implementation of the "one vs all" multi-class classifier, using scipy.optmise 'TNC' algorithm for the logistic linear regression.

Input: 
      -Initial input. Comma-separated-values (csv) formatted training data. The code is written assuming the target variables are in row 0 of the data set  
The code checks the data and displays 10 random training examples  
The code then performs oneVsAll classification, outputting the optimised parameter for logistic regression.
      -Second input. The code then asks for Comma-separated-values (csv) formatted test data, it randomly selects 10 values to test. 
 The code displays the 10 randomly chosen test examples then performs classifiers the examples, and prints out a vector of the classes of the data.      

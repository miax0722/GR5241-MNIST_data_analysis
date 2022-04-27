------------------------Instructions and Explanation of Project---------------------------

This project is for hand written digits processing, and all questions shown in the requirement are distributed in order in the final project file.

Below are instructions and explanation of codes

***NOTE: Questions are not shown directly with a question number, and the distribution of questions are shown below:

Project 1: MNIST data - Handwritten Digits Processing
	O. Background
	I. Import data and splitting 
	II. EDA --- Q1(a)-(c)
	III. Before Deep Learning --- Q2 (a), (b)
	IV. Deep Learning
		(i) 1-layer neural network --- Q3 (a)-(d)
		(ii). CNN with one layer of 2-D Convolution and pooling Q4
		(iii). CNN with three times of convolution Q5
	V. More about Deep Learning Q6, Q7
	VI. Conclusion

##########################################################################################################
# I. Import data and splitting 

In this part we import necessary packages and import data from tensorflow server. All functions and packages import from `sklearn` are for machine learning in part III, and all functions and packages from tensorflow or keras are for deep learning in Part IV and V.

##########################################################################################################
#II. EDA

In this part we plot a sample from training data and scale all X for further analysis, and this is because we want to analyze image data in [0,1]. We also print the shape of our data and do the one-hot encoding for response variable here and summarize the advantage of this method.

##########################################################################################################
#III. Before Deep Learning

In this part we tried to implement 4 machine learning methods including KNN, tree based Adaboost, Decision Tree and SVM with Gaussian Kernel. all attributes mentioned above are actually implemented as attributes in functions from sklearn. 

We also set a timer for each algorithm for comparison, and calculate the error of each algorithm. In this part, we also do the grid search cross validation for best parameter sets in each algorithm, but we omit the code for KNN and Adaboost, and for the sake of saving time we commented all parameter tuning codes.

After comparing these 4 algorithms, we also referred random forest for improvement of decision tree and do hyperparameter tuning for svm in order to have a better prediction performance.

For your reference, the codes are written in similar structures.

##########################################################################################################
#IV. Deep Learning

##(i) 1-layer neural network

In this part we implement a 1-layer Neural Network for the same dataset. The 1 layer NN has a form like (784*1->100*1->10*1), so we use Sequential() to create this neural network and add Dense(100)and Dense(10) behind it, while Dense(100) is the hidden layer. 

Besides, we set SGD as the optimizer with learning rate=0.1, loss as SparseCategoricalCrossentropy(), which stands for cross entropy for multiple labels, and metrics as either SparseCategoricalCrossentropy() or 'accuracy' for prediction accuracy.

After selecting random seed and the epoch times, we use training data and validation data, which is segmented from the original data as 6:1, to train the model and use test set to evaluate the model performance.

After selecting best metric and random seed, we visualize the weight of the hidden layer. Then we do hyperparameter tuning for learning rate and momentum by two for loops. The standard for best model is the model with smallest error on test set, i.e. smallest test error.

##(ii) CNN with one time convolution

Again we will redo the steps mentioned above, but for the sake of time we only set seed=4 and epoch=20 in case of overfitting. One thing we first need to do is reshape the X to n*28*28*1 so we can do the following convolution and maxpooling. 

Then we design the 1 time convolution CNN model. Still, we use a Sequential to create CNN model, and use a Conv2D() with filters=32, padding way ='same' (keeps the out put size = input size), kernel_size=(3,3) and activation function=reLU. Then we do the MaxPooling with pool_size = (2,2) to reduce the dimension of feature maps. Then, same as above, we do the flatten() to flatten the features in vector and project all features to a n*10 array.

Still, we set SGD as the optimizer with learning rate=0.1, loss as SparseCategoricalCrossentropy(), and metrics as either SparseCategoricalCrossentropy() or 'accuracy' for prediction accuracy, and mainly use accuracy as our metrics to select the best model later.

We see when epoch goes to around 10, the error rate becomes stable, but in parameter tuning we still set epoch=20, and parameter set is {'lr':[0.01,0.1,0.2,0.3], 'momentum':[0.1,0.3,0.5]}. After parameter tuning, we also draw the weights from best algorithm.

##(iii) CNN with three times convolution

Again we will redo the steps mentioned above with same seed=4 and epoch=20.

Then we design the 3 times convolution CNN model. Still, we use a Sequential to create CNN model, and use three Conv2D() with filters=32, 64 and 64 and other same attributes. Then we do the MaxPooling with pool_size = (2,2) and add BatchNormalization() to avoid overfitting. Finally use flatten() to flatten the features in vector and project all features to a n*10 array.

The optimizer, parameter tuning process and weight visualization are the same as above.

##########################################################################################################
#V. More about Deep Learning

In this part, we'll analyze the problem of arithmatic in digit recognition. We first read and reshape the data, and figure out samples order in row The final digit of each sample is the label -- the sum of two digits, and the first 1568 columns are the pixels of 28*56 images.

Then we applied CNN with 1 and 3 time(s) of convolution in this problem. First we reshape and scale the training, validation and test data for same reason. We'll still use grid search and same parameter set to train the model with training data train.csv and validation data val.csv, and we'll select the best algorithm that has the smallest validation error. Also, we visualized the weights.




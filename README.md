# B551 Assignment 4: Machine learning
##### Submission by Sri Harsha Manjunath - srmanj@iu.edu; Vijayalaxmi Bhimrao Maigur - vbmaigur@iu.edu; Disha Talreja - dtalreja@iu.edu
###### Fall 2019

## Image classification
In this assignment we'll study a straightforward image classification task. These days, all modern digital cameras include a sensor that detects which way the camera is being held when a photo is taken. This meta-data is then included in the image file, so that image organization programs know the correct orientation |
i.e., which way is \up" in the image. But for photos scanned in from file or from older digital cameras, rotating images to be in the correct orientation must typically be done by hand. Your task in this assignment is to create a classifier that decides the correct orientation of a given image, as shown in Figure 1.

Data. To help you get started, we've prepared a dataset of images from the Flickr photo sharing website. The images were taken and uploaded by real users from across the world, making this a challenging task on a very realistic dataset.
Since image processing is beyond the scope of this class, we don't expect you
to implement any special techniques related to images in particular. Instead, we'll simply treat the raw images as numerical feature vectors, on which we can then apply standard machine learning techniques. In particular, we'll take an n x m x 3 color image (the third dimension is because color images are stored as three separate planes { red, green, and blue), and append all of the rows together to produce a single vector of size 1 x 3mn. We've done this work for you already, so that you can treat the images simply as vectors and do not have to worry about them being images at all. Thorugh Canvas we've provided two ASCII text files, one for the training dataset and one for testing, that contain the feature vectors. To generate this file, we rescaled each image to a very tiny \micro-thumbnail" of 8 x 8 pixels, resulting in an 8 x 8 x 3 = 192 dimensional feature vector. The text files have one row per image, where each row is formatted like:

```
photo_id correct_orientation r11 g11 b11 r12 g12 b12 ...

where:
• photo id is a photo ID for the image.
• correct orientation is 0, 90, 180, or 270. Note that some small percentage of these labels may be wrong because of noise; this is just a fact of life when dealing with data from real-world sources.
• r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc., each in the range 0-255.
```


Although you can get away with just using the above text files, you may want to inspect the original images themselves, for debugging or analysis purposes, or if you want to change something about the way we've created the feature vectors (e.g. experiment with larger or smaller \micro-thumbnails"). You can view the
images in two different ways:

• You can view the original high-resolution image on Flickr.com by taking just the numeric portion of the photo id in the file above (e.g. if the photo id in the file is test/123456.jpg, just use 123456), and then visiting the following URL:

http://www.flickr.com/photo_zoom.gne?id=numeric photo id

• We'll provide a zip file of all the images in JPEG format on Canvas. We've reduced the size of each image to a 75 x 75 square, which still (usually) preserves enough information to figure out the image orientation. The ZIP file also includes UNIX scripts that will convert images in the zip file to the ASCII feature vector file format above. If you want, this lets you experiment with modifying the script to produce other feature vectors (e.g. smaller sized images, or in different color mappings, etc.) and to run your classifier on your own images.

The training dataset consists of about 10,000 images, while the test set contains about 1,000. For the training set, we've rotated each image 4 times to give four times as much training data, so there are about 40,000 lines in the train.txt file (the training images on Flickr and the ZIP file are all oriented correctly already. In test.txt, each image occurs just once (rotated to a random orientation) so there are only about 1,000 lines. If you view these images on Flickr, they'll be oriented correctly, but those in the ZIP file may not be. What to do. Your goal is to implement and test several different classfiers on this problem: k-nearest neighbors, decision trees, and neural networks. For training, your program should be run like this:

```
./orient.py train train_file.txt model_file.txt [model]
```

where [model] is one of nearest, tree, nnet, or best. This program uses the data in train _file.txt to produce a trained classifier of the specfied type, and save the parameters in model _file.txt. You may use any file format you'd like for model_file.txt; the important thing is that your test code knows how to interpret it. For testing, your program should be run like this:

```
./orient.py test test_file.txt model_file.txt [model]
```

where [model] is again one of nearest, tree, nnet, best. This program should load in the trained parameters from model file.txt, run each test example through the model, display the classication accuracy (in terms of percentage of correctly-classfied images), and output a file called output.txt which indicates the estimated label for each image in the test file. The output file should correspond to one test image per line, with the photo id, a space, and then the estimated label, e.g.:
```
test/124567.jpg 180
test/8234732.jpg 0
```
Here are more detailed specfications for each classifier.
•nearest: At test time, for each image to be classfied, the program should find the k \nearest" images in the training file, i.e. the ones with the closest Euclidean distance (or other distance, e.g. Manhattan) and have them vote on the correct orientation. (The \training" routine for this classifier will probably
just make a copy of the training data into the model file, although you can do other processing also if you prefer.) In your writeup, test and report how accuracy on the test set varies as a function of k.
• tree: Uses a decision tree to estimate the orientation. You'll need to define some binary conditions to be tested at each node. As a starting point, you might use nodes that simply compare pairs of individual pixel values, e.g. that test whether the red pixel at position 1,1 is greater than or less than the green pixel at position 3,8. Your decision tree learning process can try all possible combinations (roughly 1922) or randomly generate some pairs to try. In your report, describe how your accuracy varies as a function of the depth of the tree. A big advantage to decision trees is explainability - It's possible to understand why the classier made a given decision. If you train a very small tree (e.g., with only 3 levels), what features does it cue on?
• nnet: Implement a fully-connected feed-forward network to classify image orientation, and implement the backpropagation algorithm to train the network using gradient descent.

Note that, you have to implement these classiers from scratch. It's not allowed to use any pre-implemented packages, e.g., scikit-learn. Please check with us before using packages other than os, sys, math, matplotlib, scipy, and numpy. (Yes, we know you could get much better results by using Tensor flow or PyTorch, but that's not the point!)
Each of the above machine learning techniques has a number of parameters and design decisions. It would be impossible to try all possible combinations of all of these parameters, so identify a few parameters and conduct experiments to study their effect on the final classfication accuracy. In your report, present neatly- organized tables or graphs showing classfication accuracies and running times as a function of the parameters you choose. Which classfiers and which parameters would you recommend to a potential client? How does performance vary depending on the training dataset size, i.e. if you use just a fraction of the training data? Show a few sample images that were classfied correctly and incorrectly. Do you see any patterns to the errors?


As in the past, a small percentage of your assignment grade will be based on how accurate your "best" algorithm is with respect to the rest of the class. We will use a separate test dataset, so make sure to avoid overfitting! Although we expect the testing phase of your classifiers to be relatively fast, we will not evaluate the efficiency of your training program. Please commit your models for each method to your git repo when you submit your source code, and please name them nearest model.txt, tree model.txt, nnet model.txt, and best model.txt.


## Solutions

#### Model 1 - K Nearest Neighbours

1. Classification Accuracy & Running time as a function of parameters


![alt text](https://github.com/smanj2/Machine-Learning-Algorithms-From-Scratch/blob/master/imgs/nets2.jpeg)



2. How does the performance vary depending on the training dataset size, i.e. if you use just a fraction of the training data?


![alt text](https://github.com/smanj2/Machine-Learning-Algorithms-From-Scratch/blob/master/imgs/knn.jpg)


#### Model 2 - Neural Nets

1. Classification Accuracy & Running time as a function of parameters

The accuracies obtained below are from use a sigmoid activation function, however the same implementation was executed with Relu where slightly better results were observed. The image below provides the statistics related to Neural Nets

![alt text](https://github.com/smanj2/Machine-Learning-Algorithms-From-Scratch/blob/master/imgs/nets1.jpeg)


2. How does the performance vary depending on the training dataset size, i.e. if you use just a fraction of the training data?


![alt text](https://github.com/smanj2/Machine-Learning-Algorithms-From-Scratch/blob/master/imgs/nnets3.jpeg)

#### Model 3 - Decision Trees
1. Classification Accuracy & Running time as a function of parameters
![alt text](https://github.com/smanj2/Machine-Learning-Algorithms-From-Scratch/blob/master/imgs/decision_trees.jpg)


2. How does the performance vary depending on the training dataset size, i.e. if you use just a fraction of the training data?

Using 25% of the training data decreased the model accuracy by 5%. This is expected as the model is less likely to generalize well after learning from fewer data points

##### Additional Questions - 
1. Which model would you recommend ?

 We would recommened either among Neural Nets and KNN as we see very similar accuracy for both of them.
 
2. Here are a few examples of correct and incorrect classifications from one of the above algorithms

![alt text](https://github.com/smanj2/Machine-Learning-Algorithms-From-Scratch/blob/master/imgs/overall.jpeg)

Additionally, here is a summary table that explains the split between true and false classifiations over the entire dataset


![alt text](https://github.com/smanj2/Machine-Learning-Algorithms-From-Scratch/blob/master/imgs/summary.jpg)

#### References:

###### Decision Trees
[ 1 ] https://www.bogotobogo.com/python/scikit-learn/scikt_machine_learning_Decision_Tree_Learning_Informatioin_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php

[ 2 ] https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/


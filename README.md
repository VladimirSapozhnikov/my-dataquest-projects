# My DataQuest Python Projects
This repository contains projects that I have completed in the [DataQuest](https://www.dataquest.io/learn-with-dataquest/) bootcamp.

##  Project 1: [Analysis of apps from App Store and Google Play](https://github.com/VladimirSapozhnikov/my-dataquest-projects/blob/main/Project%201/Analysis%20of%20apps%20project.ipynb)

### Objective:
To find an optimal Category for an app developer to build a free app in, based on the data.

### Project Milestones:
- [x] Clean datasets by removing non-English apps, duplicates, incorrect data, non-free apps.
- [x] Create and analyse frequency tables for total number of apps developed in App Store and Google Play.
- [x] Hypothesise that the optimal app category is "Games" based on the analysis done so far.
- [x] Perform deeper analysis by looking at more factors such as "number of installs" and "number of reviews".
- [x] Reject the initial Hypothesis and come to the conclusion that the optimal category is "Social Networking".
      
## Project 2: [Bayes Spam Filter](https://github.com/VladimirSapozhnikov/my-dataquest-projects/blob/main/Project%202/Naive%20Bayes%20for%20message%20classification%20project.ipynb)

### Objective:
To develop a spam filter based on the multinomial ["Naive Bayes"](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier using a dataset of 5k classified sms messages.

### Project Milestones:
- [x] Explore the dataset using the [pandas](https://pandas.pydata.org/) library and plan out the designing/testing stages of the project.
- [x] Split the data into train/test sets.
- [x] Create a set of unique words from the training set (called vocabulary).
- [x] Transform the train set so that each entry contains the message, the label (spam or ham) and the number of times a word from the vocabulary has been mentioned in that particular message.
- [x] Perform the required calculations (such as the probability that a message is spam, total number of words in all spam messages etc.)                                       
- [x] Create the function that performs spam filtering based on [this](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_na%C3%AFve_Bayes) formula.
- [x] Determine the accuracy of the algorithm by performing the classification on the test set (98.83%).
- [x] Analyse messages that were given the wrong label (type 1 and type 2 errors) and identify the reason why that was the case.

## Project 3: [Predicting a Car's market price](https://github.com/VladimirSapozhnikov/my-dataquest-projects/blob/main/Project%203/Predicting%20car's%20market%20price%20using%20its%20attributes%20project.ipynb)

### Objective:
To predict a car's market price based on a rage of factors such as its make, engine size, horsepower, wheel base etc. using the KNN 
([K Nearest Neighbors Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)).

### Project Milestones:
- [x] Clean the dataset by dealing with missing data, using the [missingno](https://github.com/ResidentMario/missingno) library to decide on the best apprach.
- [x] Normalise the data so it can be used with KNN using the [pandas](https://pandas.pydata.org/) library.
- [x] Create a pipeline to train and validate simple univariate KNN models in order to begin performing hyperparameter optimisation.
- [x] Using the pipeline, decide which columns produce the most accurate univariate models with the default k-value. 
- [x] Update the pipeline so it takes into account custom k-values, and use it to test different models with k-values ranging from 1 to 9.
- [x] Visualise the results by using the [matplotlib](https://matplotlib.org/) library.
- [x] Update the pipeline to use the multivariate KNN model.
- [x] Use the pipeline to determine the combination of columns that produces the most accurate models testing it with k-values ranging from 1 to 25 and visualise those results as well.
- [x] Decide on the optimal model parameters using the results above.
- [x] Analyse the performance of the model in general, and in this particular case.

## Project 4: [Random Forest Regressor For Bike Rentals](https://github.com/VladimirSapozhnikov/my-dataquest-projects/blob/main/Project%204/Project_Predicting_Bike_Rentals.ipynb)

## Project 5: [Deep Learning Digits Classifier](https://github.com/VladimirSapozhnikov/my-dataquest-projects/blob/main/Project%205/Project_digits_classifier.ipynb)

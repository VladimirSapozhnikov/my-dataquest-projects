# My DataQuest Python Projects
This repository contains projects that I have completed in the [DataQuest](https://www.dataquest.io/learn-with-dataquest/) bootcamp.

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
- [x] Determine the accuracy of the algorithm by comparing predicted labels to the actual labels given in the dataset. 
- [x] Identify messages that were given the wrong label (type 1 and type 2 errors) and analyse them.

## Project 3: [Predicting a Car's market price](https://github.com/VladimirSapozhnikov/my-dataquest-projects/blob/main/Project%203/Predicting%20car's%20market%20price%20using%20its%20attributes%20project.ipynb)

### Objective:
To predict a car's market price based on a range of factors such as its make, engine size, horsepower, wheel base etc. using the KNN 
([K Nearest Neighbors Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)).

### Project Milestones:
- [x] Clean the dataset by dealing with the missing data, using the [missingno](https://github.com/ResidentMario/missingno) library to decide on the best approach.
- [x] Normalise the data so it can be used with KNN using the [pandas](https://pandas.pydata.org/) library.
- [x] Create a pipeline to train and validate simple univariate KNN models in order to perform hyperparameter optimisation.
- [x] Using the pipeline, select the columns that produce the most accurate univariate models with the default k-value. 
- [x] Update the pipeline so it takes into account custom k-values, and use it to test different models with k-values ranging from 1 to 9.
- [x] Visualise the results by using the [matplotlib](https://matplotlib.org/) library.
- [x] Update the pipeline to tain and validate multivariate KNN models.
- [x] Use the pipeline to determine the combination of columns that produces the most accurate models, testing it with k-values ranging from 1 to 25 and visualise those results.
- [x] Decide on the optimal model parameters using the results above.
- [x] Analyse the performance of the model in general, and in this particular case.

## Project 4: [Random Forest Regressor For Bike Rentals](https://github.com/VladimirSapozhnikov/my-dataquest-projects/blob/main/Project%204/Project_Predicting_Bike_Rentals.ipynb)

### Objective: 
To predict the total number of bikes rented in a given hour based on factors such as outside air temperature, month, day of the week (etc.) using several ml models including random forest.

### Project Milestones:
- [x] Transform the "hr" feature by making another categorical feature called "time label" 
- [x]  Perform the train/test split and move on to the model selection stage
- [x]  Discuss the usefulness of linear regression in the context of this project as well as prepare the features for ml.
- [x]  Train and test the lr model, and evaluate its perfomance in terms of mse.
- [x]  Discuss the usefulness of the [decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) model in the context of this project and comapre it to the linear regression model.
- [x]  Train and test the decision tree, and compare its accuracy to the lr model.
- [x]  Introduce the random forest algorithm, discussing its advantages and drawbacks when compared to the other ml models.
- [x]  Train and test the [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) algorithm, evaluating its performance.
- [x]  perform hyperparameter optimisation, and evaluate the improved accuracy of the predictions.

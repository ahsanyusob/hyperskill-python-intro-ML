## Hyperskill - Introductory Machine Learning in Python

Table of contents
---

- [Table of contents](#table-of-contents)
- [Challenging](#challenging)
- [Hard](#hard)
- [Medium](#medium)
- [Easy](#easy)
- [Notes](#notes)

Challenging
---

> 1. [Linear Regression from Scratch](https://hyperskill.org/projects/195?track=28) --> [my solution](https://github.com/ahsanyusob/hyperskill-python-intro-ML/blob/master/challenging/Linear%20Regression%20From%20Scratch/regression.py)  
> 
> | No | Tasks |  
> | --- | --- |  
> | 1 | Load a pandas DataFrame containing features X and target y |  
> | 2 | Create a CustomLinearRegression **CLASS** with: (1) fit, predict, r2_score and rmse **METHODS**, and (2) intercept and coefficient **ATTRIBUTES** |  
> | 3 | Instantiate CustomLinearRegression and also LinearRegression from sklearn.linear_model |  
> | 4 | Fit the data by passing features X and target y to both regression models |  
> | 5 | Predict y for the other feature datasets |  
> | 6 | Calculate RMSE and R2 metrics for both regression models |  
> | 7 | Determine the intercept and coefficient of both regression models and calculate their differences |  
>
> 2. [Logistic Regression From Scratch](https://hyperskill.org/projects/219) --> [my solution](https://github.com/ahsanyusob/hyperskill-python-intro-ML/blob/master/challenging/Logistic%20Regression%20From%20Scratch/logistic.py)
>
> | No | Tasks |  
> | --- | --- |  
> | 1 | Create a CustomLogisticRegression **CLASS** with: (1) predict_proba, sigmoid, fit_mse, fit_log_loss, and predict **METHODS**, and (2) coef_, mean_squared_error_, and log_loss_error **ATTRIBUTES** |
> | 2 | Load the Breast Cancer Wisconsin dataset from sklearn.datasets |
> | 3 | Filter and select the following features X: "worst concave points", "worst radius", and "worst perimeter" |
> | 4 | Standardize X using Z-Standardization function |
> | 5 | Split the dataset including the target variable y into training and test sets where TRAIN_SIZE=0.8 & RANDOM_STATE=43 |
> | 6 | Instantiate the CustomLogisticRegression class where FIT_INTERCEPT=True, L_RATE=0.01, and N_EPOCH=1000 |
> | 7 | Fit the model using fit_mse method with Stochastic Gradient Descent algorithm |
> | 8 | Predict the value of y_hat (predicted y) |
> | 9 | Calculate the accuracy score using sklearn.metrics.accuracy_score(y_test, y_train) |
> | 10 | Print out the accuracies & coefficient values of the logistic regression model |
> | 11 | Repeat step 6-10 for 2 more models: (1) CustomLogisticRegression using fit_log_loss method and (2) sklearn.linear_model.LogisticRegression with fit method |
> | 12 | Compare the accuracy scores for all three models as well as errors from first and last epoch of the first 2 models  |
> 
> 3. [Classification of Handwritten Digits](https://hyperskill.org/projects/205?track=28) --> [my solution (Stage1-4)](https://github.com/ahsanyusob/hyperskill-python-intro-ML/blob/master/challenging/Classification%20of%20Handwritten%20Digits/stage4.py) --> [my solution (Stage 5)](https://github.com/ahsanyusob/hyperskill-python-intro-ML/blob/master/challenging/Classification%20of%20Handwritten%20Digits/stage5.py)
>
> #### Stage 1-4
>
> | No | Tasks |  
> | --- | --- |  
> |   | Import **LIBRARIES**, **PACKAGES** or **MODULES** needed: _tensorflow_, _numpy_, _pandas_, _sklearn_, _random_ |
> |   | Implement sklearn implementations of the CLASSIFIERS and the ACCURACY SCORERS for: (1) KNN-, (2) Decision Tree, (3) Logistic Regression, and (4) Random Forest |
> |   | Import sklearn.preprocessing.Normalizer |
> | 1 | Load MNIST dataset from tensorflow.keras.datasets |
> | 2 | Using numpy, reshape the features array to the 2D array with n rows and m columns (n: number of images in the dataset; m: number of pixels in each image) --> ***FLATTEN THE FEATURES AND TARGET ARRAYS*** |
> | 3 | Use the first 6000 rows of the dataset and split it into TRAIN (70%) and TEST (30%) datasets. RANDOM_SEED is set to 40 |
> | 4 | Instantiate the Normalizer, transform the features and save the output to FEATURES_NORM |
> | 5 | Instantiate all four models and set RANDOM_STATE=40 when needed |
> | 6 | Implement separate function to make it easier for training a lot of models i.e. def fit_predict_eval(model, features_train, features_test, target_train, target_test) where model is iterated through (knn_model, dt_model, lr_model, rf_model) --> ***Undergo training for both original and normalized dataset using all FOUR models for comparison*** |
> | 7 | Fit the models |
> | 8 | Make predictions and print the accuracies in the following order: (knn, dt, logistic regression, rf) |
> | 9 | Determine whether Normalization have a positive impact in general |
> | 9 | Determine the best two models i.e. ***2 models with the best accuracy scores*** |
>
> #### Stage 5
>
> | No | Tasks |  
> | --- | --- |
> |   | Import _sklearn.model_selection.GridSearchCV_ |
> | 1 | Choose two best models for MNIST classification problem from stage 1-4 --> ***KNN and RF Classifiers*** |
> | 2 | Instantiate GridSearchCV(estimator=..., param_grid=..., scoring='accuracy', n_jobs=-1) |
> |   | 2.1 - For KNN estimator, instantiate KNeighborsClassifier() |
> |   | 2.2 - For RF estimator, instantiate RandomForestClassifier(random_state=40) |
> |   | 2.3 - For KNN parameter grid: n_neighbors=[3,4], weights=['uniform', 'distance'], algorithm=['auto', 'brute'] |
> |   | 2.4 - For RF parameter grid: n_estimators=[300, 500], max_features=['auto', 'log2'], class_weight=['balanced', 'balanced_subsample'] |
> | 3 | Run the fit method of GridSearchCV to find the best estimator |
> | 4 | Repeat Step 6-7 of stage 1-4: (a) Call fit_predict_eval function where model is iterated through (knn_best_estimator, rf_best_estimator), and (b) Fit the models. Finally, print the best sets of parameters (best estimators) for both KNN and RF algorithms and their accuracies.

Hard
---

Medium
---

Easy
---

Notes
---

> *My progress and solutions in hyperskill-introductory-machine-learning-in-python...*
>
> *Practicing github readme...*

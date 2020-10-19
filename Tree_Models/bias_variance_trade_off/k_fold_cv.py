# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)


#Note that since cross_val_score has only the option of evaluating the negative MSEs, its output should be multiplied by negative one to obtain the MSEs. The CV RMSE can then be obtained by computing the square root of the average MSE.

# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, 
                                  scoring='neg_mean_squared_error', 
                                  n_jobs=-1) 

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))


# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

#comparing RMSE train and CV RMSE we can predict whether the data has high variance or high bias. Also from baseline RMSE (baseline_RMSE which corresponds to the root mean-squared error achieved by the regression-tree trained with the disp feature only 
# (it is the RMSE achieved by the regression tree trained) which serves as the baseline RMSE above which a model is considered to be underfitting and below which the model is considered 'good enough'.

#dt suffers from high variance when RMSE_CV is far less than RMSE_train.

#dt suffers from high bias when RMSE_CV ≈ RMSE_train and both scores are greater than baseline_RMSE.

#dt is a good fit when RMSE_CV ≈ RMSE_train and both scores are smaller than baseline_RMSE.
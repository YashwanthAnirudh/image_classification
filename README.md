# image_classification

# Problem statement 
The goal is to develop a machine learning model that can accurately classify images of five specific sports persons. This involves creating a robust image classification pipeline, optimizing model performance through hyperparameter tuning, and saving the trained model for integration with a Flask server to enable real-time predictions on the web platform.

# Process
**1. Data Collection:**

Manually downloaded images of five sports celebrities and stored them in the images_dataset folder.

**2. Data Cleaning**

Objective: Retain only relevant images containing a clear face with two eyes.

**Steps:**

Converted colored images to grayscale.

Used OpenCV's Haar Cascade technique (from the official documentation) to detect faces with two eyes.

Cropped the detected faces and saved them in the cropped folder for model training.

Performed manual cleaning of the cropped folder by removing irrelevant or incorrectly cropped images.

**3. Feature Engineering**

**Steps:**

Performed wavelet transformation (from Stack Overflow) to extract important features from the face.

Combined the raw image and wavelet-transformed image vertically using NumPy's stacking method to create the input dataset (X).

Created the target dataset (Y) containing the names of the sports celebrities.

Outcome: X and Y datasets were prepared for model training.

**4. Model Training**

Initial Model:

Selected Support Vector Machine (SVM) as it is effective for classification tasks.

Split the dataset into training and testing sets using train_test_split.

Scaled input data using sklearn.pipeline to ensure uniformity across images.

Trained the model and evaluated accuracy using model.score(X_test, Y_test), achieving an initial accuracy of 80%.

Generated a classification report to analyze performance.

Hyperparameter Tuning:
Applied GridSearchCV to test different models and used 5-fold cross-validation to identify the best model and its validation score.

When you use grid search cv , it will divide the xtrain and y train  in to 2 parts (i.e training data and validation data) . when you check for the score/accuracy , it will give the score on validation data

**Results:**

On validation data (from GridSearchCV), achieved an accuracy of 79%.
On test data, i.e score.(xtest, ytest) achieved an accuracy of 86%.

# Outcome
Best Model: Support Vector Machine (SVM) was selected as the final model.

Model Evaluation:

Plotted the confusion matrix using Seaborn's heatmap with:
X-axis: Predicted values
Y-axis: True values.

**Model Saving:**
Saved the trained SVM model as a .pkl file using joblib for deployment.

# Conclusion
The SVM model performed effectively with an accuracy of 86% on test data. This trained model is ready for integration into a Flask server and subsequent deployment on the website.

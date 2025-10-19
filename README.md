#### 🚗 Car Price Prediction using Gradient Boosting Regressor

This project builds a machine learning model to predict used car prices based on multiple car features such as engine capacity, mileage, power, and torque. The model uses the Gradient Boosting Regressor algorithm from Scikit-learn to achieve accurate predictions after thorough data preprocessing, feature engineering, and outlier removal.

#### 🧠 Project Overview

The goal of this project is to predict car prices by leveraging regression-based ensemble learning. The notebook walks through every stage of the ML pipeline — from data cleaning to model evaluation, demonstrating a complete workflow for regression problems.

#### 🗂️ Dataset

File used: car details v4.csv

Target variable: Price

Sample features:

Year — Year of manufacture

Kilometer — Distance driven

Fuel Type — Petrol/Diesel/CNG

Transmission — Manual/Automatic

Owner — Number of previous owners

Engine, Max Power, Max Torque — Car performance specs

#### ⚙️ Workflow
1. 📥 Data Loading

Imported the dataset using Pandas and explored its structure.

#### 2. 🧹 Data Preprocessing

Handled missing values and duplicates.

Cleaned numeric columns (Engine, Max Power, Max Torque) using regex extraction.

Converted string-based numeric columns to float.

#### 3. 📊 Exploratory Data Analysis (EDA)

Visualized distributions of numeric variables using Seaborn histograms with KDE.

Detected skewness and identified potential transformations.

Removed outliers using the IQR method for columns like Price and Kilometer.

#### 4. 🔢 Feature Encoding

Used Label Encoding for categorical variables to prepare data for model training.

#### 5. 🤖 Model Building

Trained a Gradient Boosting Regressor from sklearn.ensemble.

#### Evaluated Model performance using key metrics:

MSE: 38601291044.05604

RMSE: 196472.1126370255

R^2 Score: 0.9481848064789217

#### 6. 📈 Model Evaluation

Interpreted model performance and residual errors to assess accuracy.

#### 🧩 Technologies Used

Category	Libraries
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn
Machine Learning	scikit-learn
Utilities	re (Regular Expressions)

#### 🧰 How to Run
Prerequisites

Make sure you have the following installed:

pip install pandas numpy matplotlib seaborn scikit-learn

Run the Notebook
jupyter notebook GradientBoost.ipynb

#### 📚 Key Learnings

Cleaning inconsistent numerical columns with regex.

Removing outliers using the IQR method.

Using Gradient Boosting for regression tasks.

Evaluating model performance with statistical metrics.

#### 🚀 Future Improvements

Apply Hyperparameter Tuning using GridSearchCV or RandomizedSearchCV


###### APP LINK 

https://huggingface.co/spaces/nandha-01/CarPriceUsingGradientBoosting

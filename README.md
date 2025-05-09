# Crop-Yield-Prediction and paddy disease detection-Using-Machine-Learning

## Project Summary
This project focuses on building a predictive system for agricultural crop yield and disease detection using machine learning models. By leveraging historical data on weather, pesticides, temperature, and disease symptoms, this project aims to estimate crop yields and detect potential diseases for various regions and crop types. The models developed for crop yield prediction include Linear Regression, Lasso, Ridge, Decision Tree, and K-Nearest Neighbors. Additionally, the project also includes a disease detection system for paddy crops using image classification techniques.

## Objectives
1. **Data Cleaning & Exploration:** Remove missing values, duplicates, and irrelevant columns to prepare the dataset for modeling.
   [Dataset Link](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
3. **Modeling & Prediction:** Train and evaluate several regression models, including:
   - Linear Regression
   - Lasso Regression
   - Ridge Regression
   - Decision Tree Regressor
   - K-Nearest Neighbors Regressor
4. **Performance Comparison:** Compare the performance of these models using metrics like Mean Absolute Error (MAE) and R² Score.
5. **Predictive System:** Develop a prediction system that allows users to input environmental and crop data to predict crop yield for specific regions.
6. **Save Models:** Save the best-performing models and preprocessing steps using Python's `pickle` library for future use.

## Workflow
1. **Data Preprocessing:**
   - Imported the dataset and dropped irrelevant columns.
   - Handled missing values and removed duplicates.
   - Encoded categorical features (Area and Item) using OneHotEncoder and scaled numerical features with StandardScaler.

2. **Model Training & Evaluation:**
   - Split the data into training and test sets.
   - Trained five different regression models.
   - Evaluated model performance based on MAE and R² Score.
   - Visualized the model performance for comparison.

3. **Prediction System:**
   - Created a function that allows predictions based on user inputs (year, rainfall, pesticide usage, temperature, area, and crop type).
   - Used Decision Tree and K-Nearest Neighbors as the main models for prediction.

4. **Model Saving:**
   - Saved the trained models (`dtr_model.pkl` and `knn_model.pkl`) and the preprocessing pipeline (`preprocesser.pkl`) using `pickle`.
  
## Snapshot
![model comparison](https://github.com/ramnathreddy-modiyam/Agriculture-crop-yiled-and-paddy-disease-dectection/blob/main/Assest/model%20comperasion.png)

### User Interface
![user interface](https://github.com/ramnathreddy-modiyam/Agriculture-crop-yiled-and-paddy-disease-dectection/blob/main/Assest/user-interface.png)
git add ../README.md

## Conclusion
The Decision Tree Regressor and K-Nearest Neighbors showed the most promise in accurately predicting crop yield based on historical data. For disease detection, the CNN model performed well in classifying images of paddy leaves. The project demonstrates the utility of machine learning in both crop yield prediction and disease detection, helping farmers and agronomists make informed decisions about crop management.

## Future Work
1.Model Improvement: Experiment with more advanced models such as Random Forest, Gradient Boosting, or Neural Networks to improve prediction accuracy for both crop yield and disease detection.

2.Feature Engineering: Incorporate more features such as soil quality, water usage, and fertilizer data to enhance the crop yield prediction model.

3.Deployment: Develop a web-based user interface (possibly using Streamlit) for end-users to easily input data and obtain yield predictions and disease detections in real time.

4.Time Series Analysis: Introduce time series forecasting techniques for crop yield prediction to account for trends and seasonality.

5.Expand Disease Detection: Extend the disease detection model to identify diseases in other crops beyond paddy.



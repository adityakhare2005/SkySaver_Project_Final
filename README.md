# Skysaver: Airline Flight Price Predictor

## Overview
Skysaver is a machine learning-powered flight price prediction application designed to help users estimate airfare costs based on key travel details. Built using **Streamlit**, the application predicts flight prices using a **regression model**, adjusts results with **airline-specific premium markups**, and evaluates whether the price is higher than usual by comparing it against historical averages for Economy and Business flights.

## Features
- **Flight Price Prediction:** Estimates prices based on user inputs (source, destination, departure/arrival times, class, airline, stops, days till travel).
- **Dynamic Input Handling:** Ensures the destination cannot be the same as the source city.
- **Airline-Specific Premium Adjustments:** Applies markups based on airline selection.
- **Historical Price Evaluation:** Checks whether the predicted price is higher than the average for Economy/Business flights.
- **Custom UI & Styling:** Streamlit interface with a **plain black theme** and an **animated toolbar featuring a plane takeoff GIF**.

## Project Architecture
1. **User Input:** Flight details entered via Streamlit UI.
2. **Data Preprocessing:** One-hot encoding of categorical features.
3. **Model Prediction:** Regression model predicts base price.
4. **Airline Premium Adjustment:** Adjusts prediction using pre-defined airline markups.
5. **Historical Price Evaluation:** Compares predicted price with class-specific averages.
6. **Final Output:** Displays estimated price along with a message indicating whether the price is above usual levels.

## Technologies Used
- **Python**
- **Streamlit** (Frontend UI)
- **Scikit-learn** (Machine Learning)
- **Pandas** (Data Handling)
- **Joblib** (Model Storage)
- **CSS** (Custom Styling)

skysaver/
│── app.py                   # Main application script
│── models/
│   ├── price_predictor.pkl   # Trained regression model
│   ├── encoder.pkl           # Feature encoding model
│── data/
│   ├── original_categories.csv  # Flight data categories
│   ├── flight_data.csv          # Historical flight prices
│── README.md                 # Project documentation



## Requirements
-**streamlit**
-**pandas**
-**numpy**
-**scikit-learn**
-**joblib**

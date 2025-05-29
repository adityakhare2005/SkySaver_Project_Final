import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Airline Flight Price Predictor", 
    page_icon="✈", 
    layout="centered"
)

# --- Minimal CSS Styling for a Plain Black Theme & Hiding Menu/Footer ---
custom_css = """
<style>
    /* Hide the Streamlit hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Set a plain black background and white text */
    body {
        background-color: #000000;
        color: #ffffff;
        font-family: Arial, sans-serif;
    }
    /* Header styling */
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        color: #ffffff;
    }
    /* Subheader styling */
    .subheader {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 30px;
        color: #ffffff;
    }
    /* Result box styling */
    .result-box {
        padding: 20px;
        background-color: #333333;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
        margin-top: 30px;
    }
    /* Styling for select boxes and slider labels */
    .stSelectbox label, .stSlider label {
        font-weight: 600;
        color: #ffffff;
        font-size: 1.1rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Page Title and Tagline ---
st.markdown("<div class='header'>Airline Flight Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Find the best deals for your next flight!</div>", unsafe_allow_html=True)

# --- Load Model & Artifacts ---
model, feature_names = joblib.load("D:/4th sem/Elective Project/Project_11/models/price_predictor.pkl")
original_df = pd.read_csv("D:/4th sem/Elective Project/Project_11/data/original_categories.csv")
encoder = joblib.load("D:/4th sem/Elective Project/Project_11/models/encoder.pkl")

# --- Layout for Input Fields Organized into Rows ---

# Row 1: Source and Destination
col1, col2 = st.columns(2)
with col1:
    source = st.selectbox("Source City", sorted(original_df['source'].unique()))
with col2:
    destination_options = sorted([dest for dest in original_df['destination'].unique() if dest != source])
    destination = st.selectbox("Destination City", destination_options)

# Row 2: Departure Time and Arrival Time side-by-side
col1, col2 = st.columns(2)
with col1:
    departure_time = st.selectbox("Departure Time", sorted(original_df['departure_time'].unique()))
with col2:
    arrival_time = st.selectbox("Arrival Time", sorted(original_df['arrival_time'].unique()))

# Row 3: Class and Airline
col1, col2 = st.columns(2)
with col1:
    travel_class = st.selectbox("Class", sorted(original_df['class'].unique()))
with col2:
    airline = st.selectbox("Airline", sorted(original_df['airline'].unique()))

# Row 4: Stops and Days till Travel (Days limited to 1-100)
col1, col2 = st.columns(2)
with col1:
    stops = st.selectbox("Stops", sorted(original_df['stops'].unique()))
with col2:
    days_till_travel = st.slider("Days till travel", 1, 100)

# --- Assemble Input DataFrame ---
input_df = pd.DataFrame(
    [[source, destination, travel_class, departure_time, arrival_time, airline, stops, days_till_travel]],
    columns=["source", "destination", "class", "departure_time", "arrival_time", "airline", "stops", "days_till_travel"]
)

# --- Encode Categorical Features ---
encoded_input = encoder.transform(
    input_df[['source', 'destination', 'departure_time', 'arrival_time', 'class', 'airline', 'stops']].to_numpy()
)
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out())

# Remove original categorical columns and merge with the encoded ones
input_df = input_df.drop(columns=['source', 'destination', 'departure_time', 'arrival_time', 'class', 'airline', 'stops']).reset_index(drop=True)
input_df = pd.concat([input_df, encoded_df], axis=1)

# --- Ensure Input Features Align with the Trained Model ---
missing_cols = set(feature_names) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[feature_names]

# --- Make a Prediction ---
predicted_price = model.predict(input_df)[0]

# --- Adjust Price Based on Airline Premium ---
airline_markup = {
    "Air_India": 0.18,
    "GO_FIRST": 0.10,
    "Indigo": 0.07,
    "Vistara": 0.25,
    "SpiceJet": 0.03,
    "AirAsia":0.00
}
markup = airline_markup.get(airline, 0)  # Default markup 0 if airline not in dictionary.
final_price = predicted_price * (1 + markup)

# If the final price is negative, remove the minus sign
if final_price < 0:
    final_price = abs(final_price)


# --- Adjust Price Based on Airline Premium ---
markup = airline_markup.get(airline, 0)
final_price = predicted_price * (1 + markup)

if final_price < 0:
    final_price = abs(final_price)

# --- Adjust Price Based on Airline Premium ---
markup = airline_markup.get(airline, 0)
final_price = predicted_price * (1 + markup)
if final_price < 0:
    final_price = abs(final_price)

# --- Incorporate Class-Specific AI-based Price Evaluation ---
try:
    # Load historical flight data which should include a "price" and "class" column.
    flight_data = pd.read_csv("D:/4th sem/Elective Project/Project_11/data/flight_data.csv")
    
    # Calculate average prices for each class.
    # Ensure that the 'class' values in your flight_data match those from your input.
    economy_avg = flight_data[flight_data['class'].str.lower() == "economy"]['price'].mean()
    business_avg = flight_data[flight_data['class'].str.lower() == "business"]['price'].mean()
    
    # Use the travel_class from the input to select the appropriate average.
    if travel_class.lower() == "business":
        avg_price = business_avg
    else:
        avg_price = economy_avg
    
    # Compare the final price to the appropriate average.
    if final_price > avg_price:
        price_message = f"This price is higher than usual for {travel_class} flights."
    else:
        price_message = f"This price is within the usual range for {travel_class} flights."
except Exception as e:
    price_message = ""
st.markdown(f"<div class='result-box'>Estimated Flight Price: ₹{final_price:,.2f}<br>{price_message}</div>", unsafe_allow_html=True)

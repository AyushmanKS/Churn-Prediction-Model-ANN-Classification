import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# --- Load Model and Encoders ---
# These are loaded once when the app starts.
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# --- Streamlit App Interface ---
st.title('Live Customer Churn Prediction')
st.write("Adjust the parameters to see the churn probability update in real-time.")

# --- User Input ---
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    credit_score = st.slider('Credit Score', 300, 850, 650)
    age = st.slider('Age', 18, 92, 38)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)

with col2:
    balance = st.slider('Balance', 0, 250000, 125000)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
    is_active_member = st.selectbox('Is Active Member?', [0, 1])
    estimated_salary = st.slider('Estimated Salary', 0, 200000, 100000)


# --- Live Prediction Logic ---
# This block runs every time a user changes an input widget.

# 1. Create a DataFrame from the user's base inputs.
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# 2. One-hot encode the 'Geography' feature.
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# 3. Combine the base and encoded data.
combined_df = pd.concat([input_df, geo_encoded_df], axis=1)

# 4. IMPORTANT: Ensure columns are in the same order as the training data.
#    You must adjust this list to match the exact order your model was trained on.
expected_column_order = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]
final_df = combined_df.reindex(columns=expected_column_order)

# 5. Scale the data and make a prediction.
input_data_scaled = scaler.transform(final_df)
prediction_prob = model.predict(input_data_scaled)[0][0]


# --- Display Live Probability ---
st.metric(label="Churn Probability", value=f"{prediction_prob:.2%}",
          delta=f"{prediction_prob - 0.5:.2%}", delta_color="inverse")


# --- Final Prediction on Button Click ---
# This block only runs when the user clicks the button.
if st.button('Show Final Prediction', type="primary"):
    if prediction_prob > 0.5:
        st.error('The Customer is LIKELY to churn.', icon="🚨")
    else:
        st.success('The Customer is NOT likely to churn.', icon="✅")
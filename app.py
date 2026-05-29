import streamlit as st
import pandas as pd
import joblib

# 1. Load the AI Brain
model = joblib.load('iris_model.pkl')

# 2. Design the App
st.title("🌸 Iris Flower Predictor")
st.write("Adjust the petal and sepal measurements below, and the AI will predict the exact species of the flower!")

# 3. Create Inputs in the Sidebar
st.sidebar.header("Flower Measurements")
# The format is: (Label, minimum, maximum, default value)
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# 4. Predict Button
if st.button("Predict Species 🌼"):
    
    # Format the data exactly how the AI expects it (matching the 4 columns)
    user_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    
    # 5. Make the Prediction
    # The model returns a 0, 1, or 2. 
    prediction = model.predict(user_data)[0]
    
    # Map those numbers back to the actual flower names
    species_map = {
        0: "Setosa", 
        1: "Versicolor", 
        2: "Virginica"
    }
    
    final_species = species_map[prediction]
    
    # 6. Display the Result
    st.markdown("---")
    st.success(f"### The AI predicts this is an **Iris {final_species}**!")
    st.balloons()
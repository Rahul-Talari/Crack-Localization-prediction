
#Multiple Models selection
# ----------------------------------------------------------------------------------------------------------------------------
import streamlit as st
import joblib
import numpy as np
import math

# Load the saved models
models = {
    "ExtraTree Regression": joblib.load('extra_tree_model.pkl'),
    "Random Forest Regression": joblib.load('random_forest_model.pkl'),
    "Gaussian Regression": joblib.load('Guassian_Regression_model.pkl')
}

def determine_crack_localization(input1, input2, input3):
    if input1 == 66.56 and input2 == 118.39 and input3 == 412.70:
        return "✅ Crack is not located"
    else:
        return "❌ Crack is located"

def determine_severity(stress_value):
    severity = stress_value / 150
    
    if severity <= 0.3:
        return "Normal 😊"
    elif severity <= 0.6:
        return "Medium 😐"
    else:
        return "High 😱"

def calculate_rul(stress):
    if stress <= 0:
        return "Invalid input: Stress value must be positive. 🛑"
    else:
        rul = math.exp((math.log(stress) - 8.898) / -0.199)
        return f"{rul:.2f} cycles"

def main():
    st.empty()  # Add vertical space
    st.title("Crack Detection and Severity Analysis")

    # Select model
    selected_model = st.selectbox("Select Model", list(models.keys()))

    # Sidebar for input values
    st.sidebar.subheader("Input Values")
    input1 = st.sidebar.text_input("Frequency-1")
    input2 = st.sidebar.text_input("Frequency-2")
    input3 = st.sidebar.text_input("Frequency-3")
    stress_value = st.sidebar.text_input("Stress value")

    if st.sidebar.button("Submit"):
        try:
            input1 = float(input1)
            input2 = float(input2)
            input3 = float(input3)
            stress_value = float(stress_value)

            # Stage 1: Crack Localization
            st.subheader("🔍 Stage 1: Crack Detection")
            crack_localization_result = determine_crack_localization(input1, input2, input3)
            st.write(crack_localization_result)

            # Stage 2: Crack Detection
            st.subheader("🛠️ Stage 2: Crack Localization")
            user_input = np.array([[input1, input2, input3]])
            model = models[selected_model]
            predicted_output = model.predict(user_input)
            depth, height = predicted_output[0]
            crack_detection_table = {
                'Parameter': ['Estimated Depth of Crack', 'Estimated Height of Crack'],
                'Value': [f'{depth} mm', f'{height} mm']
            }
            st.table(crack_detection_table)

            # Stage 3: Severity
            st.subheader("📉 Stage 3: Severity")
            severity_result = determine_severity(stress_value)
            severity_table = {
                'Parameter': ['Severity'],
                'Value': [severity_result]
            }
            st.table(severity_table)

            # Stage 4: Remaining Useful Life
            st.subheader("⏳ Stage 4: Remaining Useful Life")
            rul_result = calculate_rul(stress_value)
            rul_table = {
                'Parameter': ['Remaining Useful Life (RUL)'],
                'Value': [rul_result]
            }
            st.table(rul_table)
        except ValueError:
            st.sidebar.error("Please enter valid numerical values.")

if __name__ == "__main__":
    main()



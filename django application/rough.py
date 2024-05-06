import joblib
import os 
from django.shortcuts import render
import math

# Define the directory containing the machine learning models
current_dir = os.path.dirname(os.path.abspath(__file__))
models = os.path.join(current_dir, 'models')

# Load the machine learning models
crack_model = joblib.load(os.path.join(models, 'gaussian_model.joblib'))

def signup(request):
    return render(request, 'signup.html')

def login(request):
    return render(request, 'login1.html')

def home(request):
    if request.method == 'POST':
        # Get input values from the POST request
        f1 = float(request.POST.get('frequency_1'))
        f2 = float(request.POST.get('frequency_2'))
        f3 = float(request.POST.get('frequency_3'))
        stress_value = float(request.POST.get('stress_value'))

        # Check if the beam is healthy or not based on specific frequencies
        if f1 == 66.691 and f2 == 119.46 and f3 == 428.34:
            prediction = "Healthy Beam!" 
        else:
            prediction = "Crack Detected!" 
        		
        # HEALTHY BEAM 
        prediction_class = "green-text"
        length = "NO"
        depth = "NO"
        sif = 0 
        severe = "NA" 
        life = "Ambient"

        # UNHEALTHY BEAM 
        if prediction == "Crack Detected!":
            prediction_class = "red-text"            
            # Apply the crack model to get length and depth
            predictions = crack_model.predict([[f1, f2, f3]])
            length, depth = round(predictions[0][0], 2), round(predictions[0][1], 2)

            # Apply the stress model to get the stress intensity factor (SIF) & Determine severity based on SIF
            sif = round(stress_value / 150, 2)
            if 0 < sif <= 0.2:
                severe = "Low severity" 
            elif 0.2 < sif <= 0.5:
                severe = "Medium severity" 
            elif sif > 0.5:
                severe = "High severity" 

            # Estimate remaining life
            life = round(math.exp((math.log(stress_value) - 8.898) / -0.199), 2)

        context = {
            'prediction': prediction, 
            'prediction_class': prediction_class,
            'length': length,
            'depth': depth,
            'sif': sif, 
            'severe': severe,
            'life' : life 
        }

        # Render the updated page with the calculated values
        return render(request, 'index.html', context)

    return render(request, 'index.html')

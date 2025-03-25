from django.shortcuts import render
from django.http import JsonResponse
import os
import joblib
import numpy as np

def predict(request):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'ice_cream_model.pkl')
    model = joblib.load(model_path)

    prediction = None
    if request.method == 'POST':
        try:
            temperature = float(request.POST.get('temperature'))
            # Prepare the input for prediction
            features = np.array([[temperature]])
            pred = model.predict(features)[0]
            prediction = round(pred, 2)
        except Exception as e:
            prediction = f"Error: {e}"
        return JsonResponse({'prediction': prediction})
    
    return render(request, 'predict.html')

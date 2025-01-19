from django.shortcuts import render
import joblib
import os
import warnings
from django.conf import settings
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for server-side plotting
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import numpy as np

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        voting_model = joblib.load(os.path.join(settings.STATICFILES_DIRS[0], 'voting_model.joblib'))
        scaler = joblib.load(os.path.join(settings.STATICFILES_DIRS[0], 'scaler.joblib'))
        
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

def preprocess_input(data):
    """
    Convert form data to model input format
    """
    # Mapping dictionaries
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Smoker': {'Yes': 1, 'No': 0},
        'Blood_Pressure': {'High': 2, 'Normal': 1, 'Low': 0},
        'Alcohol_Consumption': {'Yes': 1, 'No': 0},
        'Heart_Rate': {'High': 2, 'Normal': 1, 'Low': 0},
        'Cholesterol': {'High': 2, 'Normal': 1, 'Low': 0},
        'Diabetes': {'Yes': 1, 'No': 0}
    }
    
    # Convert input data
    processed_data = {
        'Gender': mappings['Gender'][data['gender']],
        'Age': int(data['age']),
        'Smoker': mappings['Smoker'][data['smoker']],
        'Blood_Pressure': mappings['Blood_Pressure'][data['blood_pressure']],
        'Alcohol_Consumption': mappings['Alcohol_Consumption'][data['alcohol_consumption']],
        'Heart_Rate': mappings['Heart_Rate'][data['heart_rate']],
        'Cholesterol': mappings['Cholesterol'][data['cholesterol']],
        'Diabetes': mappings['Diabetes'][data['diabetes']]
    }
    
    # Create feature list in correct order
    features = ['Gender', 'Age', 'Smoker', 'Blood_Pressure', 'Alcohol_Consumption', 
               'Heart_Rate', 'Cholesterol', 'Diabetes']
    
    # Convert to list in correct order
    input_data = [[processed_data[feature] for feature in features]]
    
    return input_data

def make_prediction(input_data):
    """
    Make predictions using the ensemble model
    """
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Get predictions from ensemble model
    predictions = {
        'ensemble': {
            'prediction': 'Yes' if voting_model.predict(scaled_data)[0] == 1 else 'No',
            'probability': f"{voting_model.predict_proba(scaled_data)[0][1] * 100:.1f}",
            'raw_probability': voting_model.predict_proba(scaled_data)[0][1]
        }
    }
    
    return predictions

def generate_visualizations(prediction_data, input_data):
    """Generate visualizations for the prediction results"""
    visualizations = {}
    
    # Create a directory for visualizations if it doesn't exist
    viz_dir = os.path.join(settings.MEDIA_ROOT, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Risk Factors Radar Chart
    categories = ['Age', 'Blood Pressure', 'Heart Rate', 'Cholesterol', 'Risk Factors']
    # Normalize values for radar chart
    age_normalized = min(float(input_data['age']) / 100, 1)
    bp_value = {'High': 1, 'Normal': 0.5, 'Low': 0}[input_data['blood_pressure']]
    hr_value = {'High': 1, 'Normal': 0.5, 'Low': 0}[input_data['heart_rate']]
    chol_value = {'High': 1, 'Normal': 0.5, 'Low': 0}[input_data['cholesterol']]
    
    # Calculate risk factor score (0-1) based on smoking, diabetes, and alcohol
    risk_factors = (
        (1 if input_data['smoker'] == 'Yes' else 0) +
        (1 if input_data['diabetes'] == 'Yes' else 0) +
        (1 if input_data['alcohol_consumption'] == 'Yes' else 0)
    ) / 3
    
    values = [age_normalized, bp_value, hr_value, chol_value, risk_factors]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    categories = np.concatenate((categories, [categories[0]]))
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, label='Patient Values')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], size=10)
    ax.set_ylim(0, 1)
    plt.title('Patient Risk Factors Analysis', pad=20)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    visualizations['risk_factors'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 2. Health Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics = ['Blood Pressure', 'Heart Rate', 'Cholesterol']
    values = [bp_value, hr_value, chol_value]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = plt.bar(metrics, values, color=colors)
    plt.title('Health Metrics Analysis', pad=20)
    plt.ylabel('Risk Level')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}', ha='center', va='bottom')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    visualizations['health_metrics'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 3. Lifestyle Risk Factors
    plt.figure(figsize=(10, 6))
    lifestyle_factors = ['Smoking', 'Alcohol', 'Diabetes']
    lifestyle_values = [
        1 if input_data['smoker'] == 'Yes' else 0,
        1 if input_data['alcohol_consumption'] == 'Yes' else 0,
        1 if input_data['diabetes'] == 'Yes' else 0
    ]
    colors = ['#e74c3c' if val == 1 else '#2ecc71' for val in lifestyle_values]
    
    bars = plt.bar(lifestyle_factors, lifestyle_values, color=colors)
    plt.title('Lifestyle Risk Factors', pad=20)
    plt.ylabel('Present (1) / Absent (0)')
    plt.ylim(0, 1.2)
    
    for bar in bars:
        height = bar.get_height()
        status = "Present" if height == 1 else "Absent"
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                status, ha='center', va='bottom')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    visualizations['lifestyle_factors'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 4. Age-Related Risk Analysis
    plt.figure(figsize=(10, 6))
    age = int(input_data['age'])
    age_categories = ['Low Risk\n(18-35)', 'Moderate Risk\n(36-55)', 'High Risk\n(56+)']
    age_risks = [0.3, 0.6, 0.9]  # Example risk levels for age categories
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    bars = plt.bar(age_categories, age_risks, color=colors, alpha=0.3)
    current_age_category = 0 if age <= 35 else 1 if age <= 55 else 2
    
    # Highlight user's age category
    bars[current_age_category].set_alpha(1.0)
    
    plt.title('Age-Related Risk Analysis', pad=20)
    plt.ylabel('Risk Level')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add age indicator
    plt.axhline(y=min(age/100, 0.9), color='red', linestyle='--', label=f'Your Age ({age})')
    plt.legend()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    visualizations['age_analysis'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return visualizations

def index(request):
    prediction = None
    visualizations = None
    form_data = None
    
    if request.method == 'POST':
        try:
            # Process form data
            form_data = request.POST.dict()
            model_input = preprocess_input(form_data)
            
            # Make predictions
            prediction = make_prediction(model_input)
            
            # Generate visualizations
            visualizations = generate_visualizations(prediction, form_data)
            
        except Exception as e:
            prediction = {'error': str(e)}
    
    return render(request, 'index.html', {
        'prediction': prediction,
        'visualizations': visualizations,
        'form_data': form_data
    })


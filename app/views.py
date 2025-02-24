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
from django.http import JsonResponse

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load models
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        voting_model = joblib.load(os.path.join(settings.STATICFILES_DIRS[0], 'voting_model.joblib'))
        scaler = joblib.load(os.path.join(settings.STATICFILES_DIRS[0], 'scaler.joblib'))
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Global constants from hypertensiondetection.py
FEATURE_ORDER = [
    'Gender', 'Age', 'Genetic_History', 'Smoking_Habits',
    'Alcohol_Consumption', 'Cholesterol', 'Heart_Rate',
    'Diabetes', 'Systolic_BP', 'Diastolic_BP'
]

CATEGORICAL_MAPPINGS = {
    'Gender': {'male': 1, 'female': 0},
    'Genetic_History': {'yes': 1, 'no': 0},
    'Smoking_Habits': {'often': 3, 'sometimes': 2, 'rarely': 1, 'never': 0, 'always': 3},
    'Alcohol_Consumption': {'often': 3, 'sometimes': 2, 'rarely': 1, 'never': 0}
}

NUMERICAL_RANGES = {
    'Age': (18, 90),
    'Cholesterol': (0.0, 13.0),
    'Heart_Rate': (55, 130),
    'Diabetes': (0.0, 27.0),
    'Systolic_BP': (70, 200),
    'Diastolic_BP': (55, 130)
}

# Add these constants after NUMERICAL_RANGES
MEAN_VALUES = {
    'Cholesterol': 1.58,  # Mean value from the training dataset
    'Diabetes': 0.85      # Mean value from the training dataset
}

def preprocess_input(data):
    """Convert form data to model input format"""
    # Convert input data
    processed_data = {}
    
    # Handle categorical features
    processed_data['Gender'] = CATEGORICAL_MAPPINGS['Gender'][data['gender'].lower()]
    processed_data['Genetic_History'] = CATEGORICAL_MAPPINGS['Genetic_History'][data['genetic_history'].lower()]
    processed_data['Smoking_Habits'] = CATEGORICAL_MAPPINGS['Smoking_Habits'][data['smoking_habits'].lower()]
    processed_data['Alcohol_Consumption'] = CATEGORICAL_MAPPINGS['Alcohol_Consumption'][data['alcohol_consumption'].lower()]
    
    # Handle numerical features
    processed_data['Age'] = float(data['age'])
    processed_data['Heart_Rate'] = float(data['heart_rate'])
    processed_data['Systolic_BP'] = float(data['systolic_bp'])
    processed_data['Diastolic_BP'] = float(data['diastolic_bp'])
    
    # Handle optional fields with mean values
    cholesterol = data.get('cholesterol', '').strip()
    diabetes = data.get('diabetes', '').strip()
    
    processed_data['Cholesterol'] = float(cholesterol) if cholesterol else MEAN_VALUES['Cholesterol']
    processed_data['Diabetes'] = float(diabetes) if diabetes else MEAN_VALUES['Diabetes']
    
    # Create input data in correct order
    input_data = [[processed_data[feature] for feature in FEATURE_ORDER]]
    
    return input_data

def make_prediction(input_data):
    """Make predictions using the ensemble model"""
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Get prediction and probability
    prediction = voting_model.predict(scaled_data)[0]
    probability = voting_model.predict_proba(scaled_data)[0][1]
    
    # Format results
    predictions = {
        'ensemble': {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': f"{probability * 100:.1f}",
            'raw_probability': probability
        }
    }
    
    return predictions

def generate_visualizations(prediction_data, input_data):
    """Generate visualizations for the prediction results"""
    visualizations = {}
    
    # 1. Risk Factors Radar Chart
    categories = ['Age', 'Blood Pressure', 'Heart Rate', 'Cholesterol', 'Risk Factors']
    
    # Normalize values based on NUMERICAL_RANGES
    age_normalized = (float(input_data['age']) - NUMERICAL_RANGES['Age'][0]) / (NUMERICAL_RANGES['Age'][1] - NUMERICAL_RANGES['Age'][0])
    systolic_bp = float(input_data['systolic_bp'])
    bp_normalized = (systolic_bp - NUMERICAL_RANGES['Systolic_BP'][0]) / (NUMERICAL_RANGES['Systolic_BP'][1] - NUMERICAL_RANGES['Systolic_BP'][0])
    hr_normalized = (float(input_data['heart_rate']) - NUMERICAL_RANGES['Heart_Rate'][0]) / (NUMERICAL_RANGES['Heart_Rate'][1] - NUMERICAL_RANGES['Heart_Rate'][0])
    
    # Handle optional fields with mean values
    cholesterol = input_data.get('cholesterol', '')
    diabetes = input_data.get('diabetes', '')
    
    chol_value = float(cholesterol) if cholesterol.strip() else MEAN_VALUES['Cholesterol']
    diabetes_value = float(diabetes) if diabetes.strip() else MEAN_VALUES['Diabetes']
    
    chol_normalized = (chol_value - NUMERICAL_RANGES['Cholesterol'][0]) / (NUMERICAL_RANGES['Cholesterol'][1] - NUMERICAL_RANGES['Cholesterol'][0])
    
    # Calculate risk factor score
    risk_factors = (
        (float(input_data['smoking_habits'].lower() == 'often') * 0.4) +
        (float(input_data['genetic_history'].lower() == 'yes') * 0.3) +
        (float(input_data['alcohol_consumption'].lower() == 'often') * 0.3)
    )
    
    values = [age_normalized, bp_normalized, hr_normalized, chol_normalized, risk_factors]
    values = np.clip(values, 0, 1)  # Ensure values are between 0 and 1
    
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
    values = [bp_normalized, hr_normalized, chol_normalized]
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
        1 if input_data['smoking_habits'].lower() == 'often' else 0,
        1 if input_data['alcohol_consumption'].lower() == 'often' else 0,
        1 if diabetes_value > NUMERICAL_RANGES['Diabetes'][0] else 0  # Use diabetes_value instead of input_data['diabetes']
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
    age_risks = [0.3, 0.6, 0.9]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    bars = plt.bar(age_categories, age_risks, color=colors, alpha=0.3)
    current_age_category = 0 if age <= 35 else 1 if age <= 55 else 2
    
    bars[current_age_category].set_alpha(1.0)
    
    plt.title('Age-Related Risk Analysis', pad=20)
    plt.ylabel('Risk Level')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    age_normalized = min(age/100, 0.9)
    plt.axhline(y=age_normalized, color='red', linestyle='--', label=f'Your Age ({age})')
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
            
            # Validate required numerical ranges
            required_fields = ['Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP']
            for field in required_fields:
                min_val, max_val = NUMERICAL_RANGES[field]
                value = float(form_data[field.lower()])
                if not min_val <= value <= max_val:
                    raise ValueError(f"{field} must be between {min_val} and {max_val}")
            
            # Validate optional fields if provided
            optional_fields = ['Cholesterol', 'Diabetes']
            for field in optional_fields:
                value = form_data.get(field.lower(), '')
                if value and value.strip():  # Only validate if a value is provided
                    min_val, max_val = NUMERICAL_RANGES[field]
                    value = float(value)
                    if not min_val <= value <= max_val:
                        raise ValueError(f"{field} must be between {min_val} and {max_val}")
            
            # Process data and make prediction
            model_input = preprocess_input(form_data)
            prediction = make_prediction(model_input)
            
            # Generate visualizations
            visualizations = generate_visualizations(prediction, form_data)
            
            # If it's an AJAX request, return only the result section
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return render(request, 'index.html', {
                    'prediction': prediction,
                    'visualizations': visualizations,
                    'form_data': form_data,
                    'ajax_request': True
                })
            
        except ValueError as e:
            prediction = {'error': str(e)}
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            prediction = {'error': f"An error occurred: {str(e)}"}
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'error': str(e)}, status=400)
    
    return render(request, 'index.html', {
        'prediction': prediction,
        'visualizations': visualizations,
        'form_data': form_data
    })


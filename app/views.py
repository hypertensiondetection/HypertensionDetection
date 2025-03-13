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

# Global constants from hypertensiondetection.py - Updated to match the new feature order
FEATURE_ORDER = [
    'Gender', 'Age', 'Systolic_BP', 'Diastolic_BP',
    'Cholesterol', 'Heart_Rate', 'Diabetes'
]

CATEGORICAL_MAPPINGS = {
    'Gender': {'male': 1, 'female': 0},
}

NUMERICAL_RANGES = {
    'Age': (18, 90),
    'Cholesterol': (0.0, 13.0),
    'Heart_Rate': (55, 130),
    'Diabetes': (0.0, 27.0),
    'Systolic_BP': (70, 200),
    'Diastolic_BP': (55, 130)
}

MEAN_VALUES = {
    'Cholesterol': 4.53,  # Mean value from the actual dataset (4.52756)
    'Diabetes': 7.14      # Mean value from the actual dataset (7.14244)
}

def preprocess_input(data):
    """Convert form data to model input format"""
    # Convert input data
    processed_data = {}
    
    # Handle categorical features
    processed_data['Gender'] = CATEGORICAL_MAPPINGS['Gender'][data['gender'].lower()]
    
    # Handle numerical features
    processed_data['Age'] = float(data['age'])
    processed_data['Heart_Rate'] = float(data['heart_rate'])
    processed_data['Systolic_BP'] = float(data['systolic_bp'])
    processed_data['Diastolic_BP'] = float(data['diastolic_bp'])
    
    # Handle optional fields with population-specific mean values
    cholesterol = data.get('cholesterol', '').strip()
    diabetes = data.get('diabetes', '').strip()
    
    # Determine if user is likely hypertensive based on blood pressure
    is_likely_hypertensive = (processed_data['Systolic_BP'] >= 140 or 
                             processed_data['Diastolic_BP'] >= 90)
    
    # Use appropriate mean values based on likely hypertension status
    if not cholesterol:
        if is_likely_hypertensive:
            processed_data['Cholesterol'] = 4.55  # Mean for hypertensive patients
        else:
            processed_data['Cholesterol'] = 4.51  # Mean for non-hypertensive patients
    else:
        try:
            processed_data['Cholesterol'] = float(cholesterol)
        except ValueError:
            # If conversion fails, use appropriate mean
            processed_data['Cholesterol'] = 4.55 if is_likely_hypertensive else 4.51
    
    if not diabetes:
        if is_likely_hypertensive:
            processed_data['Diabetes'] = 8.78  # Mean for hypertensive patients
        else:
            processed_data['Diabetes'] = 5.52  # Mean for non-hypertensive patients
    else:
        try:
            processed_data['Diabetes'] = float(diabetes)
        except ValueError:
            # If conversion fails, use appropriate mean
            processed_data['Diabetes'] = 8.78 if is_likely_hypertensive else 5.52
    
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
    categories = ['Age', 'Blood Pressure', 'Heart Rate', 'Cholesterol']
    
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
    
    values = [age_normalized, bp_normalized, hr_normalized, chol_normalized]
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
    metrics = ['Blood Pressure', 'Heart Rate', 'Cholesterol', 'Diabetes']
    diabetes_normalized = (diabetes_value - NUMERICAL_RANGES['Diabetes'][0]) / (NUMERICAL_RANGES['Diabetes'][1] - NUMERICAL_RANGES['Diabetes'][0])
    values = [bp_normalized, hr_normalized, chol_normalized, diabetes_normalized]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
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
    
    # 3. Blood Pressure Analysis
    plt.figure(figsize=(10, 6))
    bp_categories = ['Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Crisis']
    bp_thresholds = [120, 130, 140, 180, 200]  # Systolic BP thresholds
    
    # Determine BP category
    systolic = float(input_data['systolic_bp'])
    diastolic = float(input_data['diastolic_bp'])
    
    if systolic < 120 and diastolic < 80:
        bp_category = 0  # Normal
    elif (systolic >= 120 and systolic < 130) and diastolic < 80:
        bp_category = 1  # Elevated
    elif (systolic >= 130 and systolic < 140) or (diastolic >= 80 and diastolic < 90):
        bp_category = 2  # Stage 1
    elif (systolic >= 140 and systolic < 180) or (diastolic >= 90 and diastolic < 120):
        bp_category = 3  # Stage 2
    else:
        bp_category = 4  # Crisis
    
    # Create bar colors (highlight current category)
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    alpha_values = [0.3, 0.3, 0.3, 0.3, 0.3]
    alpha_values[bp_category] = 1.0
    
    # Create bars with alpha values - Fix: Apply alpha individually to each bar
    bars = plt.bar(bp_categories, [0.2, 0.4, 0.6, 0.8, 1.0], color=colors)
    
    # Set alpha for each bar individually
    for i, bar in enumerate(bars):
        bar.set_alpha(alpha_values[i])
    
    # Add systolic/diastolic values
    plt.axhline(y=systolic/200, color='red', linestyle='--', label=f'Systolic: {systolic} mmHg')
    plt.axhline(y=diastolic/130, color='blue', linestyle='--', label=f'Diastolic: {diastolic} mmHg')
    
    plt.title('Blood Pressure Classification', pad=20)
    plt.ylabel('Severity')
    plt.ylim(0, 1.1)
    plt.legend()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    visualizations['bp_analysis'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 4. Age-Related Risk Analysis
    plt.figure(figsize=(10, 6))
    age = int(input_data['age'])
    age_categories = ['Low Risk\n(18-35)', 'Moderate Risk\n(36-55)', 'High Risk\n(56+)']
    age_risks = [0.3, 0.6, 0.9]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    # Create bars with default alpha
    bars = plt.bar(age_categories, age_risks, color=colors, alpha=0.3)
    current_age_category = 0 if age <= 35 else 1 if age <= 55 else 2
    
    # Set alpha for the current category bar
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
                    try:
                        min_val, max_val = NUMERICAL_RANGES[field]
                        value = float(value)
                        if not min_val <= value <= max_val:
                            raise ValueError(f"{field} must be between {min_val} and {max_val}")
                    except ValueError:
                        # If conversion fails, raise a more helpful error
                        raise ValueError(f"Invalid {field} value. Please enter a valid number.")
            
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


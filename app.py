from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

app = Flask(__name__)

# Render the form template
@app.route('/')
def index():
    return render_template('index.html')

# Predict using Naive Bayes model
def predict_nb(features):
    # Load the trained Naive Bayes model
    # Replace `loaded_nb_model` with your actual Naive Bayes model
    loaded_nb_model = joblib.load('nb_model.pkl')
    prediction = loaded_nb_model.predict(features)
    return prediction

# Predict using Logistic Regression model
def predict_lr(features):
    # Load the trained Logistic Regression model
    # Replace `loaded_lr_model` with your actual Logistic Regression model
    loaded_lr_model = joblib.load('lr_model.pkl')
    prediction = loaded_lr_model.predict(features)
    return prediction


# Predict using SVM model
def predict_svm(features):

    loaded_svm_model = joblib.load("svm_model.pkl")
    prediction = loaded_svm_model.predict(features)
    return prediction


# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():

    A1_Score = 1 if request.form.get('A1_Score') == 'yes' else 0
    A2_Score = 1 if request.form.get('A2_Score') == 'yes' else 0
    A3_Score = 1 if request.form.get('A3_Score') == 'yes' else 0
    A4_Score = 1 if request.form.get('A4_Score') == 'yes' else 0
    A5_Score = 1 if request.form.get('A5_Score') == 'yes' else 0
    A6_Score = 1 if request.form.get('A6_Score') == 'yes' else 0
    A7_Score = 1 if request.form.get('A7_Score') == 'yes' else 0
    A8_Score = 1 if request.form.get('A8_Score') == 'yes' else 0
    A9_Score = 1 if request.form.get('A9_Score') == 'yes' else 0
    A10_Score = 1 if request.form.get('A10_Score') == 'yes' else 0
    gender = 1 if request.form.get('gender')== 'yes' else 0
    autism = 1 if request.form.get('autism') == 'yes' else 0
    jaundice = 1 if request.form.get('jaundice') == 'yes' else 0
    ethnicity = request.form['ethnicity']
    age = request.form['age']
    country_of_res = request.form['country_of_res']
    
    country_mapping = {
    'India': 27, 'United States': 58, 'United Kingdom': 57, 'United Arab Emirates': 56,
    'New Zealand': 39, 'Mexico': 37, 'South Africa': 50, 'Romania': 45, 'Malaysia': 36,
    'Afghanistan': 0, 'Sri Lanka': 52, 'Australia': 6, 'Aruba': 5, 'Russia': 46,
    'Saudi Arabia': 47, 'Bahamas': 9, 'Netherlands': 38, 'Belgium': 11, 'Canada': 14,
    'Jordan': 34, 'France': 23, 'China': 15, 'Austria': 7, 'Iran': 29, 'Spain': 51,
    'Brazil': 13, 'Serbia': 48, 'Pakistan': 43, 'Ireland': 31, 'Armenia': 4,
    'Philippines': 44, 'Italy': 32, 'Viet Nam': 60, 'Argentina': 3, 'American Samoa': 1,
    'Hong Kong': 25, 'Germany': 24, 'Uruguay': 59, 'Azerbaijan': 8, 'Costa Rica': 16,
    'Iceland': 26, 'Kazakhstan': 35, 'Ethiopia': 21, 'Egypt': 20, 'Czech Republic': 18,
    'Tonga': 54, 'Ukraine': 55, 'Nicaragua': 40, 'Bolivia': 12, 'Japan': 33,
    'Sierra Leone': 49, 'Angola': 2, 'Oman': 42, 'Indonesia': 28, 'Finland': 22,
    'Cyprus': 17, 'Iraq': 30, 'Niger': 41, 'Bangladesh': 10, 'Sweden': 53,
    'Ecuador': 19
    }

    for name, number in country_mapping.items():
        if name == country_of_res:
            country_of_res=number  # Output: 27
            break
        
    ethnicity_mapping = {
    'Black': 2, 'South Asian': 8, 'Hispanic': 3, '?': 0, 'Latino': 4,
    'White-European': 10, 'Middle Eastern': 5, 'Asian': 1, 'Others': 6,
    'Pasifika': 7, 'Turkish': 9
    }


    for name, number in ethnicity_mapping.items():
        if name == ethnicity:
            ethnicity = number
            break

    # Convert form values to features
    features = np.array([A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, 
                          A6_Score, A7_Score, A8_Score, A9_Score, A10_Score,
                          gender, autism, jaundice, ethnicity, age, country_of_res])
    
    features=label_encoder.fit_transform(features)

    # Create a StandardScaler instance
#     scaler = StandardScaler()

# # Fit and transform the scaler on the training data
#     input_features_scaled = scaler.fit_transform(features)
    features = np.array(features).reshape(1, -1)
    # Predict using Naive Bayes model
    nb_prediction = predict_nb(features)
    
    # Predict using Logistic Regression model
    lr_prediction = predict_lr(features)
    
    # Predict using SVM model
    svm_prediction = predict_svm(features)
    
    # Return predictions
    return render_template('index.html ', nb_prediction=nb_prediction, lr_prediction=lr_prediction, svm_prediction=svm_prediction)

if __name__ == '__main__':
    app.run(debug=True)

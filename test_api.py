import requests

url = 'http://127.0.0.1:5000/predict'
# Example input: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
data = {
    'features': [2, 120, 70, 20, 79, 25.0, 0.5, 33]
}

response = requests.post(url, json=data)
print(response.json())

""" Heroku sample request """

import requests

url = "https://heroku.."
data = {'age': 37, 'workclass': 'Private', 'fnlgt': 321943, 'education': 'Prof-school', 'education_num': 15, 'marital_status': 'Married-civ-spouse', 'occupation': 'Prof-specialty',
        'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0, 'hours_per_week': 40, 'native_country': 'United-States'}
response = requests.post(url, json=data)
print(f"Status: {response.status_code}")
print(f"Result: {response.json()}")

import requests

url = 'http://127.0.0.1:5000/predict'
sample_data = {"pl_rade": 1.1, "pl_eqt": 280, "st_teff": 5700}

response = requests.post(url, json=sample_data)
print(response.json()) # This prints the Habitability result
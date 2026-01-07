import requests
import json

url = 'http://127.0.0.1:5000/predict'
headers = {'x-api-key': 'secret_api_key_123', 'Content-Type': 'application/json'}

with open('test_payload.json', 'r') as f:
    data = json.load(f)

try:
    response = requests.post(url, json=data, headers=headers)
    print(json.dumps(response.json(), indent=4))
except Exception as e:
    print(f"Error: {e}")

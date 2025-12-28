import requests

url = "http://127.0.0.1:8000/predict"
payload = {"text": "I really enjoyed this product!"}

r = requests.post(url, json=payload, timeout=10)
print("Status:", r.status_code)
print("Response:", r.json())

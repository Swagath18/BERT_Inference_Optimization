import requests

# URL of the Flask server
url = "http://127.0.0.1:5000/predict"

# Test data
data = {
    "texts": ["This is great!", "I dislike this."]
}

# Send POST request
try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
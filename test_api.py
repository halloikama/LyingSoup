import requests

result = requests.get("http://localhost:5000/")
print(result.json())
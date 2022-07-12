import json
import requests
with open("clinc_full_training_pos.txt", 'r') as f:
    data = json.load(f)

r = requests.post('http://127.0.0.1:8000/train', json=data)
print(r.json())

with open("clinc_full_testing_pos.txt", 'r') as f:
    test_data = json.load(f)

r = requests.post('http://127.0.0.1:8000/infer', json=test_data)
print(r.json())

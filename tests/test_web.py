import requests

resp = requests.post('http://localhost:5000/predict',
                     files={'file': open('blues.00000.wav', 'rb')})

print(resp.text)

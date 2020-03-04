import requests

url = 'http://localhost:5000/predict_api'
#r = requests.post(url,json={'Jira_Summary':"abc"})
r = requests.post(url,json={'Jira_Summary':2000})

print(r.json())
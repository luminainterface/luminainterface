import requests

URL = "http://localhost:9000/respond"

def test_respond():
    resp = requests.post(URL, json={"query": "What is artificial intelligence?"})
    print("Status:", resp.status_code)
    print("Response:", resp.json())

if __name__ == "__main__":
    test_respond() 
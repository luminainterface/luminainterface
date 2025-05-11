import requests

CONCEPT_DICTIONARY_URL = "http://localhost:8828"

# Example: fetch all concepts
resp = requests.get(f"{CONCEPT_DICTIONARY_URL}/concepts")
if resp.status_code == 200:
    concepts = resp.json()
    print(f"Fetched {len(concepts)} concepts.")
    if concepts:
        # Fetch the first concept by term
        term = concepts[0]['term']
        print(f"Fetching concept: {term}")
        detail_resp = requests.get(f"{CONCEPT_DICTIONARY_URL}/concepts/{term}")
        if detail_resp.status_code == 200:
            print("Concept details:")
            print(detail_resp.json())
        else:
            print(f"Failed to fetch concept {term}: {detail_resp.status_code}")
else:
    print(f"Failed to fetch concepts: {resp.status_code}") 
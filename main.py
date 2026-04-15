import json

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

 

 

 

 

with open("destinations.json", "r", encoding="utf-8") as f:

    destinations = json.load(f)

 

 

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

 

def get_embedding(text):

    return model.encode(text)

 

 

for d in destinations:

    d["embedding"] = get_embedding(d["Description"])

 

print("Vector size:", len(destinations[0]["embedding"]))

 

 

def find_similar(user_input):

    input_embedding = get_embedding(user_input)

 

    scores = []

    for d in destinations:

        score = cosine_similarity(

            [input_embedding],

            [d["embedding"]]

        )[0][0]

 

        scores.append((d, score))

 

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:3]

 

 

query = "mountain hiking"

results = find_similar(query)

 

print("\nResults for:", query)

for r in results:

    print(f"{r[0]['Name']} ({r[0]['Category']}) -> {r[1]:.3f}")
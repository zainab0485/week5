import json
import numpy as np
from sentence_transformers import SentenceTransformer
from db import SessionLocal
from models import Destination

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def to_bytes(arr):
   return np.array(arr, dtype=np.float32).tobytes()

with open("destinations.json", "r", encoding="utf-8") as f:
   data = json.load(f)

db = SessionLocal()

for item in data:
   text = f"{item['Name']} {item['Description']} {item['Category']}"
   embedding = model.encode(text)

   dest = Destination(
       name=item["Name"],
       country=item["Country"],
       description=item["Description"],
       category=item["Category"],
       embedding=to_bytes(embedding)
   )

   db.add(dest)

db.commit()
db.close()

print(" Data inserted successfully!")
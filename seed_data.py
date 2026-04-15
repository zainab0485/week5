import json
import numpy as np
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from db import engine

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def array_to_bytes(arr):
   return np.array(arr, dtype=np.float32).tobytes()


def seed():
   connection = engine.connect()


   with open("destinations.json", "r") as f:
       destinations = json.load(f)

   for dest in destinations:
       embedding = model.encode(dest["description"])
       embedding_bytes = array_to_bytes(embedding)

       connection.execute(
           text("""
               INSERT INTO destinations (name, country, category, description, embedding)
               VALUES (:name, :country, :category, :description, :embedding)
           """),
           {
               "name": dest["name"],
               "country": dest["country"],
               "category": dest["category"],
               "description": dest["description"],
               "embedding": embedding_bytes
           }
       )

   connection.commit()
   connection.close()

   print(" Data inserted successfully!")
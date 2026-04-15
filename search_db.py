import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from db import engine

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def bytes_to_array(binary_data):
   return np.frombuffer(binary_data, dtype=np.float32)


def search_destinations(user_query, top_k=3):
   connection = engine.connect()


   query_embedding = model.encode(user_query)
   query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

   result = connection.execute(text("SELECT * FROM destinations"))
   destinations = result.fetchall()

   results = []

   for dest in destinations:
      

       dest_embedding = bytes_to_array(dest[5]).reshape(1, -1)

       score = cosine_similarity(query_embedding, dest_embedding)[0][0]

       results.append({
           "name": dest[1],
           "country": dest[2],
           "category": dest[3],
           "description": dest[4],
           "score": score
       })

   connection.close()

   results = sorted(results, key=lambda x: x["score"], reverse=True)


   if results and results[0]["score"] < 0.3:
       print(" No strong match found, showing general recommendations...\n")

   return results[:top_k]


if __name__ == "__main__":
   user_query = input("Enter your travel preference: ")
   recommendations = search_destinations(user_query)

   print("\nTop Recommendations:")
   for i, rec in enumerate(recommendations, 1):
       print(f"{i}. {rec['name']} - {rec['country']} ({rec['category']})")
       print(f"   Score: {rec['score']:.4f}")
       print(f"   {rec['description']}\n")
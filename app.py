import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from db import SessionLocal
from models import Destination

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def bytes_to_array(binary_data):
   return np.frombuffer(binary_data, dtype=np.float32)

def search_destinations(user_query, top_k=3):
   db = SessionLocal()

   query_embedding = model.encode(user_query)
   query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

   destinations = db.query(Destination).all()

   results = []
   for dest in destinations:
       dest_embedding = bytes_to_array(dest.embedding).reshape(1, -1)
       score = cosine_similarity(query_embedding, dest_embedding)[0][0]

       results.append({
           "name": dest.name,
           "country": dest.country,
           "category": dest.category,
           "description": dest.description,
           "score": score
       })

   db.close()

   results = sorted(results, key=lambda x: x["score"], reverse=True)

   if results and results[0]["score"] < 0.3:
       st.warning("No strong match found, showing general recommendations.")

   return results[:top_k]

st.title("AI Travel Recommendation App")
st.write("Enter your travel preference and get the top 3 recommendations.")

user_query = st.text_input("Enter your travel preference:")

if st.button("Search"):
   if user_query.strip():
       recommendations = search_destinations(user_query)

       st.subheader("Top Recommendations")
       for i, rec in enumerate(recommendations, 1):
           st.write(f"**{i}. {rec['name']} - {rec['country']} ({rec['category']})**")
           st.write(f"Score: {rec['score']:.4f}")
           st.write(rec["description"])
           st.write("---")
   else:
       st.warning("Please enter something first.")
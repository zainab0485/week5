import streamlit as st
from search_db import search_destinations

st.title("AI Travel Recommendation App")
st.write("Enter your travel preference and get the top 3 recommendations.")


user_query = st.text_input("Enter your travel preference:")


if st.button("Search"):
   if user_query.strip():

      
       results = search_destinations(user_query)

       st.subheader("Top Recommendations")

       for i, rec in enumerate(results, 1):
           st.write(f"**{i}. {rec['name']} - {rec['country']} ({rec['category']})**")
           st.write(f"Score: {rec['score']:.4f}")
           st.write(rec["description"])
           st.write("---")

   else:
       st.warning("Please enter something.")
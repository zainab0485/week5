from sqlalchemy import Column, Integer, String, LargeBinary
from db import Base

class Destination(Base):
   __tablename__ = "destinations"

   id = Column(Integer, primary_key=True, index=True)
   name = Column(String, nullable=False)
   country = Column(String, nullable=False)
   description = Column(String, nullable=False)
   category = Column(String, nullable=False)
   embedding = Column(LargeBinary, nullable=False)
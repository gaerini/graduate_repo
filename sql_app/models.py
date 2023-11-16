from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, LargeBinary, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    password = Column(String)



class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="items")

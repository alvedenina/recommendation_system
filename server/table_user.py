from server.database import Base, SessionLocal
from sqlalchemy import Column, Integer, String, func


class User(Base):
    __tablename__ = "user"
    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    id = Column(Integer, primary_key=True, nullable=False)
    os = Column(String)
    source = Column(String)

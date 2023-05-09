from server.database import Base, SessionLocal
from sqlalchemy import Column, Integer, String


class Post(Base):
    __tablename__ = 'post'
    id = Column(Integer, primary_key=True, nullable=False)
    text = Column(String)
    topic = Column(String)

    def __repr__(self):
        return f"{self.id} - {self.topic}"   # return a machine readable representation of a type

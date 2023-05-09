from server.database import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from server.table_post import Post
from server.table_user import User


class Feed(Base):
    __tablename__ = 'feed_action'
    action = Column(String, primary_key=True, nullable=False)  # nullable=False - data can't be NULL
    post_id = Column(Integer, ForeignKey(Post.id), primary_key=True, nullable=False)
    post = relationship(Post)
    time = Column(DateTime, primary_key=True, nullable=False)
    user_id = Column(Integer, ForeignKey(User.id), primary_key=True, nullable=False)
    user = relationship(User)

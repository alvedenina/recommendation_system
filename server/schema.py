import datetime
from pydantic import BaseModel
from typing import List


class UserGet(BaseModel):
    age: int
    city: str
    country: str
    exp_group: int
    gender: int
    id: int
    os: str
    source: str

    class Config:
        orm_mode = True   # get data with all relationship data


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class FeedGet(BaseModel):
    action: str
    post_id: int
    post: PostGet
    time: datetime.datetime
    user_id: int
    user: UserGet

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

import pandas as pd
from sqlalchemy import create_engine, MetaData, update, Table, Integer, insert
from sqlalchemy.orm import Session
import os

database = os.getenv('DATABASE')
engine = create_engine(database, echo=False)
meta = MetaData()

def execute_func(update_stmt):
    with engine.connect() as con: 
        con.execute(update_stmt)
        con.commit()

def update_users(engine=engine, meta=meta):
    users = Table('users', meta, autoload_with=engine)

    update_stmt = (
                update(users).where(users.c.name == 'Pinkey').values(last_training_date=int(0))
                )   
    execute_func(update_stmt)


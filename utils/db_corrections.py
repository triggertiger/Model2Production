import pandas as pd
from sqlalchemy import create_engine, MetaData, update, Table, Integer, insert, text
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

def update_pass_hash(engine=engine, meta=meta):
    users = Table('users', meta, autoload_with=engine)
    pinkey = os.getenv('PINKEYHASH')
    print(pinkey)
    brain = os.getenv('BRAINHASH')
    print(brain)
    update_stmt1 = (
                update(users).where(users.c.name == 'Pinkey').values(password_hash=str(pinkey))
                )  
    update_stmt2 = (
                update(users).where(users.c.name == 'Brain').values(password_hash=brain)
                )
    execute_func(update_stmt1)
    execute_func(update_stmt2)

def add_datetime(engine=engine, meta=meta):
    transactions = Table('transactions', meta, autoload_with=engine)
    if 'time_stamp_datetime' not in transactions.columns:
            add_column = 'ALTER TABLE transactions ADD COLUMN time_stamp_datetime DATETIME;'
            set_content = 'UPDATE transactions SET time_stamp_datetime = DATETIME(time_stamp);'
            execute_func(text(add_column))
            execute_func(text(set_content))


if __name__=="__main__":
    #update_users()
    add_datetime()
    #update_pass_hash()

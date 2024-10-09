import sqlalchemy
from sqlalchemy import Column, Integer, String, Float, Table, create_engine
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase
from docker_config import DATABASE_FULL_PATH
import os
engine = create_engine(os.getenv('DATABASE'), echo=True)

class Base(DeclarativeBase):
    pass

meta = Base.metadata
meta.reflect(bind=engine)
    
Session = sessionmaker(bind=engine)
session = Session()
cc_transactions = meta.tables['cc_transactions']

transactions = Table(
    'transactions',
    meta,
    Column('id', Integer, primary_key=True),
    Column('time_stamp', String(50)),
    Column('user', Integer),
    Column('card', Integer),
    Column('year', Integer),
    Column('month', Integer),
    Column('day', Integer),
    Column('amount', Float),
    Column('use_chip', String(30)),
    Column('merchant_name', Integer),
    Column('merchant_city', String(30), nullable=True),
    Column('merchant_state', String(30), nullable=True),
    Column('zip', Integer, nullable=True),
    Column('mcc', Integer, nullable=True),
    Column('errors', String(255), nullable=True),
    Column('is_fraud', String(30)),
    Column('hour', Integer),
    Column('minute', Integer)
) 

meta.create_all(engine)

select = sqlalchemy.select(
    cc_transactions.c.id, 
    cc_transactions.c.time_stamp, 
    cc_transactions.c.user, 
    cc_transactions.c.card,
    cc_transactions.c.year,
    cc_transactions.c.month,
    cc_transactions.c.day,
    cc_transactions.c.amount,
    cc_transactions.c.use_chip,
    cc_transactions.c.merchant_name,
    cc_transactions.c.merchant_city,
    cc_transactions.c.merchant_state,
    cc_transactions.c.zip,
    cc_transactions.c.mcc,
    cc_transactions.c.errors,
    cc_transactions.c.is_fraud,
    cc_transactions.c.hour,
    cc_transactions.c.minute,
    )
insert = sqlalchemy.insert(transactions).from_select(
    names=['id', 'time_stamp', 'user', 'card', 'year', 'month', 'day', 'amount',
       'use_chip', 'merchant_name', 'merchant_city', 'merchant_state', 'zip',
       'mcc', 'errors', 'is_fraud', 'hour', 'minute'],
    select=select
)

with engine.connect() as conn:
    res = conn.execute(insert)
    conn.commit()
session.commit()

cc_transactions.drop(engine, checkfirst=True)

res = session.query(transactions).limit(5).all()
for li in res: 
    print(li)


from sqlalchemy import create_engine, MetaData, update, select, Table, text
import os
from dotenv import load_dotenv
load_dotenv('.env')
database = os.getenv('DATABASE')
print(database)
engine = create_engine(database, echo=False)
meta = MetaData()
print('printing engine: ', meta)
def execute_func(engine, update_stmt):
    with engine.connect() as con: 
        con.execute(update_stmt)
        con.commit()

def update_users(engine=engine, meta=meta):
    """returns the count of last training date to zero"""
    users = Table('users', meta, autoload_with=engine)

    update_stmt = (
                update(users).where(users.c.name == 'Pinkey').values(last_training_date=int(0))
                )   
    execute_func(engine, update_stmt)
    select_stmt = select(users).where(users.c.name == 'Brain')

    with engine.connect() as conn:
            res = conn.execute(select_stmt).all()
            print(res)

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
    execute_func(engine, update_stmt1)
    execute_func(engine, update_stmt2)

def add_datetime(database=database, engine=engine, meta=meta):
    large_engine = create_engine(database, isolation_level="AUTOCOMMIT")
    """adds a datetime format to the table for processing by sql_data_queries module"""
    transactions = Table('transactions', meta, autoload_with=engine)
    if 'time_stamp_datetime' not in transactions.columns:
        add_column = 'ALTER TABLE transactions ADD COLUMN time_stamp_datetime TIMESTAMP;'
        set_content = 'UPDATE transactions SET time_stamp_datetime = time_stamp::timestmap;'
        execute_func(large_engine, text(add_column))
        execute_func(large_engine, text(set_content))

def add_primary_key(database=database, engine=engine, meta=meta):
    """adds p.k to the transactions table, for further reading by sqlalchemy"""
    transactions = Table('transactions', meta, autoload_with=engine)
    add_key = 'alter table transactions add primary key(id)'
    execute_func(engine, text(add_key))
        
    

if __name__=="__main__":
    add_primary_key()
    add_datetime()
    update_pass_hash()


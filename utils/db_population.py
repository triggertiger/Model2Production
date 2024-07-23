from sqlalchemy import create_engine, MetaData, ForeignKey, text
from sqlalchemy import String
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase, Mapped, mapped_column

from typing import Optional
import pandas as pd
import os
from utils.config import DATABASE, DATA_FILE, DATA_PATH, TRAIN_DATES, USERS

# database engine:
engine = create_engine(DATABASE, echo=False)

# create a dataframe, reset the index from datetime index to continous int
fileToRead = os.path.join(DATA_PATH, DATA_FILE)
df = pd.read_csv(fileToRead)
df.reset_index(inplace=True)
df.rename(columns={'index': 'time_stamp'}, inplace=True)
df.reset_index(inplace=True)
df.rename(columns={'index': 'id'}, inplace=True)

#test df:
short = df[[ 'user', 'amount']][:15]
short.reset_index(inplace=True)
short.rename(columns={'index': 'id'}, inplace=True)
print(df.columns)
# convert to sqlite (no primary key generatead with .to_sql())
df.to_sql(
    'cc_transactions', 
    engine,
    index=False, 
    #dtype=short.dtypes,
    schema=None, 
    method='multi',  
    chunksize=10000,
    if_exists='replace')

# test select for transactions:
with engine.connect() as conn:
   res = conn.execute(text("SELECT * FROM cc_transactions LIMIT 5"))
for row in res:
    print(row)

metadata = MetaData()
metadata.create_all(engine)

# create table structures for users and dates:
class Base(DeclarativeBase):
    pass

class TrainDates(Base):
    __tablename__ = 'training_dates'
    id: Mapped[int] = mapped_column(primary_key=True)
    train_date: Mapped[str] = mapped_column(String(30))
    #users: Mapped[List['User']]= relationship()

    def __repr__(self):
        return f'TrainDate(id={self.id}, train_dates={self.train_date})'
 
class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] =mapped_column(primary_key=True)
    
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(64))
    last_training_date: Mapped[int] = mapped_column(ForeignKey('training_dates.id'))

    def __repr__(self):
        return f'User(id={self.id}, name={self.name}, password=secret, last_training_date={self.last_training_date})'

# create all the tables 
Base.metadata.create_all(engine)

# Open a session with the db:
session = Session(engine)

# insert data into tables: 
new_users = []
for u in USERS:
    name = u['name']
    password_hash = u['password']
    user = User(name=name, password_hash=password_hash, last_training_date=0)
    new_users.append(user)

session.add_all(new_users)
session.commit()
 
dates = TRAIN_DATES['dates'].to_numpy()
new_dates = [TrainDates(train_date=str(d)[:10]) for d in dates]
session.add_all(new_dates)  
session.commit()
session.close()

# test query:
Session = sessionmaker(bind=engine)
db_session = Session()
for item in db_session.query(User.id, User.name):
    print(item)

for item in db_session.query(TrainDates.id, TrainDates.train_date):
    print(item)

    
    

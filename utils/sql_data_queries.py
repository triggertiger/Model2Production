import pandas as pd
from sqlalchemy import create_engine, MetaData, update, Table, select, Integer
from sqlalchemy.orm import Session
import os
from datetime import datetime


# get data from sql database: 
class TrainDatesHandler:
    def __init__(self, date=None, username="Pinkey", DATABASE = 'DATABASE_FULL_PATH'):

# data engine:
        self.database = os.getenv(DATABASE)
        self.engine = create_engine(self.database, echo=False)
        self.dates_table = 'training_dates'
        self.users_table = 'users'
        self.transactions_table = 'transactions'
        self.username = username
        self.prediction_start_date = date
        

    @property 
    def dates_df(self):
        query = f'SELECT * FROM {self.dates_table}'
        return pd.read_sql(query, self.engine)
        

    @property 
    def last_train_date_index(self):
        query = f'SELECT name, last_training_date FROM {self.users_table} WHERE name == "{self.username}"'
        users_df = pd.read_sql(query, self.engine)
        return users_df['last_training_date'][0]
        
    
    @property 
    def last_training_date(self):#dates_df, last_training_index):
        return self.dates_df['train_date'][self.last_train_date_index]
            
    def update_db_last_train_date(self):
        
        # set connection to users table:
        meta = MetaData()

        # data to update: new dates_df index for the next date
        new_training_index = self.last_train_date_index + 1
        print(f'new_training_index: {new_training_index}')
        
        users = Table('users', meta, autoload_with=self.engine)
        update_stmt = (
            update(users).where(users.c.name == self.username).values(last_training_date=int(new_training_index) )
            )
        select_stmt = select(users).where(users.c.name == self.username)
                
        with self.engine.connect() as conn:
            conn.execute(update_stmt)
            res = conn.execute(select_stmt).all()
            print(res)
            conn.commit()
    
    def get_transactions_to_date(self):
        last_data_date = pd.Timestamp(self.last_training_date) + pd.DateOffset(months=1)
        query = f'SELECT * FROM {self.transactions_table};'
        df = pd.read_sql(query, self.engine)
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        df = df.loc[df['time_stamp'] < last_data_date]

        return df

    def get_prediction_data(self, datestring='2019-01-01'):
        
        if self.prediction_start_date:
            if isinstance(self.prediction_start_date, datetime):
                print('already datetime')
                date = self.prediction_start_date
            else: 
                date = datetime.strptime(self.prediction_start_date, '%Y-%m-%d') 
                    
        else:
            date = datetime.strptime(datestring, '%Y-%m-%d') 
            
        
        year = date.year
        month = date.month 

        query = f'SELECT * FROM {self.transactions_table} WHERE year = {year} and month = {month};'        
        df = pd.read_sql(query, self.engine)
        return df
    

if __name__ == "__main__":
    user_data = TrainDatesHandler()
    #print(user_data.last_training_date)
    #print(user_data.last_train_date_index)
    df = user_data.get_prediction_data()
    print(df.head())
    print(df.shape)

    #user_data.update_db_last_train_date()

        



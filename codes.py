import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
from itertools import product
import os
import pymysql

class Data:
    
    def __init__(self, df = None):
        self.df = df
        
    def preprocessing(self, train_data = True):
        if train_data:
            df = self.df.copy()
        else:
            df = self.test_df.copy()

        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
        df['IsHolidayInt'] = [int(x) for x in list(df.IsHoliday)]
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['week_day'] = df['Date'].dt.weekday
        df = df.drop('Date', axis=1)    
        if train_data == True:
            self.feature_df = df[df.loc[ :, df.columns != 'Weekly_Sales'].columns] 
            self.target_df = df['Weekly_Sales']
            return self.feature_df, self.target_df
        else:
            return df
        
    def read_from_csv(self,path, delimiter =','):
        df = pd.read_csv(path, delimiter=delimiter)
        df.Date = df.Date.astype('datetime64[ns]')
        df.IsHoliday = df.IsHoliday.astype('bool')
        self.df = df
        return self.df
    
    def read_from_sql_db(self,cnx, table = 'sales'):
        cursor = cnx.cursor()
        cursor.execute('select * from {}'.format(table))
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday'])
        df.Date = df.Date.astype('datetime64[ns]')
        df.IsHoliday = df.IsHoliday.astype('bool')
        self.df = df
        cnx.close()
        return self.df
    
    def write_to_sql_db(self, cnx, query=None, path=None, csv_file = True):
        cursor = cnx.cursor()
        if csv_file:
            df = pd.read_csv(path).astype('str')
            query = ("INSERT INTO sales "
               "(Store, Dept, Date, Weekly_Sales, IsHoliday) "
               "VALUES (%s, %s, %s, %s, %s)")
            cursor.executemany(query, df.to_numpy().tolist())
            cnx.commit()
        else:
            cursor.execute(query)
            cnx.commit()
        cnx.close()
        
    def test_preprocessing(self):
        date_data = self.df.Date.unique()
        date_data.sort()
        prediction_date = date_data[-1] + np.timedelta64(7,'D') # determine prediction date
        store_data = self.df.Store.unique()
        dept_data = self.df.Dept.unique()
        store_dept = list(product(store_data,dept_data))
        test_df = pd.DataFrame(store_dept, columns = ['Store', 'Dept'])
        test_df['Date'] = prediction_date
        test_df['IsHoliday'] = False
        self.test_df = test_df
        self.test_feature_df = self.preprocessing(train_data=False)
        return self.test_feature_df
    
    def create_plot_df(self, upto_month = 3, Store=1, Dept=1):
        trunc_train_df = self.df[self.df.Date >= (self.df.iloc[-1,:].Date - np.timedelta64(upto_month,'M'))]
        self.plot_df = pd.concat([trunc_train_df,self.prediction_df], axis=0)
        return self.plot_df

'''   
    def plot(self, upto_month = 3, Store=1, Dept=1):
        df = self.plot_df[(self.plot_df.Store == Store) & (self.plot_df.Dept == Dept)]
        df1 =df[df.Date<=(df.Date.max()-np.timedelta64(7,'D'))]
        df2 = df[df.Date>=(df.Date.max()-np.timedelta64(7,'D'))]
        plt.figure(figsize=(12,8))
        plt.grid()
        plt.title('Weekly Sales Prediction')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.text(1,1,'Store:{} \n Dept:{}'.format(Store, Dept))
        plt.plot(df1.Date, df1.Weekly_Sales, c='b')
        plt.plot(df2.Date, df2.Weekly_Sales, '--', c='r')
        plt.savefig('plot.png')
        return plt.figure()
'''
            
class modelling:
    def __init__(self, other):
        self.data = other
    def fit(self):
        input = [('scale', StandardScaler()), ('model', RandomForestRegressor(n_estimators = 150, n_jobs = 2, max_features = 7))]
        pipe = Pipeline(input)
        self.model = pipe.fit(self.data.feature_df, self.data.target_df)
        return self.model
    def predict(self):
        self.data.test_predicted_df = self.model.predict(self.data.test_feature_df)
        self.data.prediction_df = pd.concat([self.data.test_df,pd.DataFrame(self.data.test_predicted_df, columns=['Weekly_Sales'])], axis =1)
        return self.data.prediction_df
    

def refitting():

    db_user = os.environ.get('CLOUD_SQL_USERNAME')
    db_password = os.environ.get('CLOUD_SQL_PASSWORD')
    db_name = os.environ.get('CLOUD_SQL_DATABASE_NAME')
    db_connection_name = os.environ.get('CLOUD_SQL_CONNECTION_NAME')

        # When deployed to App Engine, the `GAE_ENV` environment variable will be
        # set to `standard`
    if os.environ.get('GAE_ENV') == 'standard':
        # If deployed, use the local socket interface for accessing Cloud SQL
        unix_socket = '/cloudsql/{}'.format(db_connection_name)
        cnx = pymysql.connect(user=db_user, password=db_password,
                                unix_socket=unix_socket, db=db_name)
    else:
        # If running locally, use the TCP connections instead
        # Set up Cloud SQL Proxy (cloud.google.com/sql/docs/mysql/sql-proxy)
        # so that your application can use 127.0.0.1:3306 to connect to your
        # Cloud SQL instance
        host = '127.0.0.1'
        cnx = pymysql.connect(user='root', password='Pixel', host=host, db='Forecast')

    data = Data()
    data.read_from_sql_db(cnx)
    data.preprocessing()
    data.test_preprocessing()
    model = modelling(data)
    model.fit()
    model.predict()
    model.data.create_plot_df()
    joblib.dump(model.data.plot_df,'plot_df.obj')
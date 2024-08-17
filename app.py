from flask import Flask, request, flash, jsonify, redirect, url_for, render_template, session
from flask_marshmallow import Marshmallow
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from flask_restful import Resource, Api, fields, reqparse
from data_prep_pipeline import predict_pipeline
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
from utils.config import DATABASE_FULL_PATH
from sqlalchemy.ext.automap import automap_base
#from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, registry
from sqlalchemy import Column, MetaData, Table
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, length, ValidationError
from datetime import datetime
import json
import pandas as pd
from utils.sql_data_queries import TrainDatesHandler
from dash import Dash, html, dash_table, dcc, callback, Input, Output



# starting Flask app and db connection
app = Flask(__name__)
app.secret_key = '123'
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_FULL_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)
app.app_context().push()
bcrypt = Bcrypt(app)
api = Api(app)
# ns = api.namespace('ns')
# ma = Marshmallow(app)

# handling login
# 1. flask form
class LoginForm(FlaskForm):
    username = StringField(
        validators=[InputRequired(), length(min=2, max=20)],
        render_kw={"placeholder": "username"})
    password = StringField(
        validators=[InputRequired(), length(min=2, max=20)],
        render_kw={"placeholder": "Password"})
    submit = SubmitField('Log In')

# 2. Setting Loginmanager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return db.session.query(Users).get(int(user_id))

# there is a conflict with sqlalchemy database reflect and the requirements of the Flask login module. 
# therefore need to set Users instance manually
# 3. setting class schema 
class Users(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    password_hash = db.Column(db.String(64), nullable=False)
    last_training_date = db.Column(db.Integer)

ReflectedBase = automap_base()
ReflectedBase.prepare(autoload_with=db.engine)
Transactions = ReflectedBase.classes.transactions
Dates = ReflectedBase.classes.training_dates
# Users = ReflectedBase.classes.users

@app.route('/')
def home():
    title = "Fraud prediction portal, welcome"
    return render_template("index.html", title=title)


@app.route("/login", methods=['POST', 'GET'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        
        #user = Users.query.filter_by(name=form.username.data).first()
        user_logging_in = db.session.query(Users).filter_by(name = form.username.data).first()
        
        if user_logging_in:
            if bcrypt.check_password_hash(user_logging_in.password_hash, form.password.data):
                flash('Logged in successfully.')
                login_user(user_logging_in)
                return redirect(url_for('userpage'))
            
    flash('wrong username or password')
    return render_template("login.html", form=form)
    
@app.route('/logout', methods=["POST", "GET"])
@login_required
def logout():
     logout_user()
     return redirect(url_for("login"))


@app.route("/userpage", methods=["POST", "GET"])
@login_required
def userpage():

    username = current_user.name
    
    last_date_index = current_user.last_training_date + 1       
    last_train_date_row = db.session.query(Dates).filter(Dates.id==last_date_index).first()
    datestring = last_train_date_row.train_date
    
    dt = datetime.strptime(datestring, '%Y-%m-%d')
    
    session['last_train_date'] = dt
    session['predict_month'] = dt.month
    session['predict_year'] = dt.year
    
    # the index number will be also the serial number of the version to be used for prediction
    # this is in order to use the relevant model version for each user "progress" in time.
    # if the mlflow model 'latest' version is smaller than this index, the model should not re-train 
    session['model_version'] = last_date_index


    return render_template('userpage.html', username=username, last_train_date=datestring)



dash_app = Dash(__name__, server=app, url_base_pathname='/predict_next/')

def predict_next(session=session):
    date = session['last_train_date']
    model_version = session['model_version']
    results = predict_pipeline(date=date, model_version=model_version)
    frauds = results[results['is_fraud']]
    return frauds

@app.route("/predict_next", methods=["POST", "GET"])
@login_required
def serve_dash_table():
    frauds_df = predict_next()
    print('starting prediction')
    dash_app.layout = [
        html.Div([
            html.H1(
                children=f'Suspicious Transactions for the period: {session["predict_month"]}/{session["predict_year"]}',
                
                style={'textAlign': 'center'}
            ),
            html.Br(),
        ]),
        html.Div([
            html.Div(children=[dash_table.DataTable(
                data = frauds_df.to_dict('records')
            )])
        ])
    ]

        
    return dash_app.index()


    

# hashing existing password:
@app.route('/test')
def test():
    
    return ' hello world'
if __name__ =="__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
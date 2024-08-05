from dash import Dash, html, dash_table, dcc, callback, Input, Output, ctx
#import dash_bootstrap_components as dbc
import pandas as pd
from data_prep_pipeline import FraudDataProcessor, load_saved_model, predict, load_model_weights, model_trainer

# set data: 
data = FraudDataProcessor()
pred_date = pd.to_datetime(data.last_training_date) + pd.DateOffset(months=1)


# month of prediction:
month = pred_date.month
year = pred_date.year


app = Dash()#(external_stylesheets=[dbc.themes.BOOTSTRAP])#__name__, template_folder="templates")
app.layout = [
    html.Div([
        html.H1(
            children=f'Suspicious Transactions for the month: {month}/{year}',
            style={'textAlign': 'center'}),
        html.Hr(),          #horizontal line
        dcc.Dropdown(['all', 'True', 'False'], 'True', id='transaction-type-dropdown'), #dropdown - show only fraud
        html.Br(),
    ]),
    html.Div([
        # design side pane
        html.Div([
            # button predict:
            html.Div([
                html.Button('Predict', id='predict_button', n_clicks=0),
            ], 
            style={'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'height': '300px',
                'margin-left': '50%',
                }),
        ],
        style={
            'position': 'fixed', 
            'top': 128, 
            'left': 0, 
            'bottom': 0, 
            'width': '20%', 
            'background-color': '#f0f0f0'}
        ),
        # design table:
        html.Div(children=[dash_table.DataTable(
            data=data.pred_df[:100].to_dict('records'),
            columns=[{'id': c, 'name': c} for c in data.pred_df.columns], 
            id='preds_tbl',
            page_size= 50,
            style_table={'overflowX': 'auto'},
            style_cell={
            # all three widths are needed
            'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            },
        )], 
        style={
            'width': '80%',
            'height': '50%',
            'border': '0.5px solid black',
            'margin': '0 auto',
            'margin-left': '20%'
        }
        ),
        # design example for callback
        html.H6(children=['here comes the pred table']),
        html.Br(),
        html.Hr(),
        html.Div(id='predictions')
    ]),
]

def get_model():
    model = load_saved_model()
    return model

def get_prediction_data_query():
    pass

@callback(
    Output(component_id='predictions', component_property='children'),
    Input(component_id='predict_button', component_property='n_clicks')
)
def predict_period(n_clicks):
    if n_clicks:
        my_data = FraudDataProcessor()
        my_data.x_y_generator()
        model = get_model()
        ful_results = predict(model, my_data)
        results_df = ful_results[:15]
        return html.Div(dash_table.DataTable(
            data=results_df.to_dict('records')
    ))

def create_df():
    pass

def dash_present_df():
    pass

def dash_present_only_fraud():
    pass

def update_database():
    # this would be my api
    # send data over api: user, last date
    pass
def retrain_next_month():
    # this would be the automated google function
    pass


if __name__ == '__main__':
    app.run(debug=True)





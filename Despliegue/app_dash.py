import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
from dotenv import load_dotenv # pip install python-dotenv
import os
import keras
import psycopg2
import pandas as pd
import os
import plotly.express as px
import numpy as np

columnas = ['Saldo_Limite', 'Sexo', 'Edad', 'Cuenta_Sept', 'Cuenta_Ago',
       'Cuenta_Jul', 'Cuenta_Jun', 'Cuenta_May', 'Cuenta_Abr', 'Pago_Sept',
       'Pago_Ago', 'Pago_Jul', 'Pago_Jun', 'Pago_May', 'Pago_Abr',
       'StatusFinal', 'Deuda', 'Educacion_Bachillerato', 'Educacion_Posgrado',
       'Educacion_Pregrado', 'Estado_Civil_Casado', 'Estado_Civil_Soltero',
       'Status_Sept_-1', 'Status_Sept_0', 'Status_Sept_1', 'Status_Sept_2',
       'Status_Sept_3', 'Status_Sept_4', 'Status_Sept_5', 'Status_Sept_6',
       'Status_Sept_7', 'Status_Sept_8', 'Status_Ago_-1', 'Status_Ago_0',
       'Status_Ago_1', 'Status_Ago_2', 'Status_Ago_3', 'Status_Ago_4',
       'Status_Ago_5', 'Status_Ago_6', 'Status_Ago_7', 'Status_Ago_8',
       'Status_Jul_-1', 'Status_Jul_0', 'Status_Jul_1', 'Status_Jul_2',
       'Status_Jul_3', 'Status_Jul_4', 'Status_Jul_5', 'Status_Jul_6',
       'Status_Jul_7', 'Status_Jul_8', 'Status_Jun_-1', 'Status_Jun_0',
       'Status_Jun_1', 'Status_Jun_2', 'Status_Jun_3', 'Status_Jun_4',
       'Status_Jun_5', 'Status_Jun_6', 'Status_Jun_7', 'Status_Jun_8',
       'Status_May_-1', 'Status_May_0', 'Status_May_1', 'Status_May_2',
       'Status_May_3', 'Status_May_4', 'Status_May_5', 'Status_May_6',
       'Status_May_7', 'Status_May_8', 'Status_Abr_-1', 'Status_Abr_0',
       'Status_Abr_1', 'Status_Abr_2', 'Status_Abr_3', 'Status_Abr_4',
       'Status_Abr_5', 'Status_Abr_6', 'Status_Abr_7', 'Status_Abr_8']

dic_civil = {'Casado':'Estado_Civil_Casado',"Soltero": 'Estado_Civil_Soltero'}
dic_educacion = {'Posgrado': 'Educacion_Posgrado','Pregrado': 'Educacion_Pregrado','Bachillerato': 'Educacion_Bachillerato'}
dic_estado = {"No uso tarjeta":-2, "Pago minimo":0, "Pago total":-1, "1M Tarde":1, "2M Tarde":2,"3M Tarde":3,"4M Tarde":4,"5M Tarde":5,"6M Tarde":6}
orden_meses = ['Status_Sept_','Status_Ago_','Status_Jul_','Status_Jun_','Status_May_','Status_Abr_']
dic_meses = {"Septiembre": 'status_sept', "Agosto":'status_ago',"Julio":'status_jul',"Junio":'status_jun',"Mayo":'status_may',"Abril":'status_abr'}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# configurar para el env.
# path to env file
env_path="C:/Users/mparr/OneDrive - Universidad de los Andes/01_Semestres/7 Semestre/ANALITICA COMPUTACIONAL/Proyecto 2ACTD/env/app.env"
# load env 
load_dotenv(dotenv_path=env_path)
# extract env variables
USER=os.getenv('USER')
PASSWORD=os.getenv('PASSWORD')
HOST=os.getenv('HOST')
PORT=os.getenv('PORT')
DBNAME=os.getenv('DBNAME')

engine = psycopg2.connect(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)

cursor = engine.cursor()
query = """
SELECT educacion, esdefault, count(esdefault)
FROM visualizar
GROUP BY educacion, esdefault;
"""
df = pd.io.sql.read_sql_query(query,engine)
# Create a bar graph
fig = px.histogram(df, x='educacion', y='count', color="esdefault", barmode='group',
            labels={'count': 'Count', 'esdefault': 'Default'})
fig.update_layout(title='Count of Defaults by Education',
                xaxis_title='Education',
                yaxis_title='Count')

cursor = engine.cursor()
query3 = """
SELECT esdefault, SUM(cuenta_sept) + SUM(cuenta_ago) + SUM(cuenta_jul) + SUM(cuenta_jun) + SUM(cuenta_may) + SUM(cuenta_abr)AS deuda_total
FROM visualizar
GROUP BY esdefault;
"""
df3 = pd.io.sql.read_sql_query(query3,engine)
fig2 = px.pie(df3, values="deuda_total", names="esdefault", title='Deuda total en default')

#cargar archivo de disco
model = keras.models.load_model("Modelos/modelo_P2.h5")

# Cargar datos de una posible predicción
app.layout = html.Div(
    [
    html.Br(),
    html.H4('Distribución de educación'),
    dcc.Graph(figure = fig),
    html.Br(),
    html.H4('Default por mes'),
    html.Div(["Mes de revisión: ",
              dcc.Dropdown(id='mes-revisar', value='Septiembre', options=['Septiembre','Agosto','Julio','Junio','Mayo','Abril'])], style={"width": "50%"}),
    dcc.Graph(id = "graph"),
    html.Br(),
    html.H4('Deuda total en default'),
    dcc.Graph(figure = fig2),
    html.Br(),
    html.H6("Ingrese las catacterísticas del cliente"),
    html.Div(["Sexo: ",
              dcc.Dropdown(id='sexo-1', value='H', options=['H', 'M'])], style={"width": "50%"}),
    html.Br(),
    html.Div(["Edad: ",
              dcc.Input(id='edad-1', type='number', value=18, placeholder='Ingrese edad', min=18, max=120)]),
    html.Br(),
    html.Div(["Educación: ",
              dcc.Dropdown(id='educacion-1', value='Bachillerato', options=['Posgrado','Pregrado','Bachillerato','Otros'])], style={"width": "50%"}),
    html.Br(),
    html.Div(["Estado Civil: ",
              dcc.Dropdown(id='estado-civil', value='Soltero', options=['Casado','Soltero','Otro'])], style={"width": "50%"}),
    html.Br(),
    html.Div(["Saldo Limite de cuenta: ",
              dcc.Input(id='saldo-limite', type='number', value=0, placeholder='Ingrese saldo limite', min=0)]),
    html.Br(),
    html.Div([
    html.Table([
        html.Tr([html.Th('Mes'), html.Th('Deuda'), html.Th('Pagado'), html.Th('Estado')]),
        *[html.Tr([
            html.Td(f'Mes {i}'),
            dcc.Input(id=f'deuda-{i}', type='number', min=0, value=0),
            dcc.Input(id=f'pago-{i}', type='number', min=0, value=0),
            dcc.Dropdown(id=f'estado-{i}', value="Pago minimo", options = ["No uso tarjeta", "Pago minimo", "Pago total", "1M Tarde", "2M Tarde","3M Tarde","4M Tarde","5M Tarde","6M Tarde"])
        ]) for i in range(1, 7)]
    ])]),
    html.H6("Probabilidad de que el cliente realice default:"),
    html.Br(),
    html.Div(["Probabilidad:", html.Div(id='output')])
    ]
)

@app.callback(
    Output(component_id='output', component_property='children'),
    [Input(component_id='sexo-1', component_property='value'),
    Input(component_id='edad-1', component_property='value'),
    Input(component_id='educacion-1', component_property='value'),
    Input(component_id='estado-civil', component_property='value'),
    Input(component_id='saldo-limite', component_property='value'),
    Input(component_id='deuda-1', component_property='value'),
    Input(component_id='pago-1', component_property='value'),
    Input(component_id='estado-1', component_property='value'),
    Input(component_id='deuda-2', component_property='value'),
    Input(component_id='pago-2', component_property='value'),
    Input(component_id='estado-2', component_property='value'),
    Input(component_id='deuda-3', component_property='value'),
    Input(component_id='pago-3', component_property='value'),
    Input(component_id='estado-3', component_property='value'),
    Input(component_id='deuda-4', component_property='value'),
    Input(component_id='pago-4', component_property='value'),
    Input(component_id='estado-4', component_property='value'),
    Input(component_id='deuda-5', component_property='value'),
    Input(component_id='pago-5', component_property='value'),
    Input(component_id='estado-5', component_property='value'),
    Input(component_id='deuda-6', component_property='value'),
    Input(component_id='pago-6', component_property='value'),
    Input(component_id='estado-6', component_property='value')]
)
def update_output_div(sexo_1,edad_1,educacion_1,estado_civil,saldo_limite,deuda_1,pago_1,estado_1,deuda_2,pago_2,estado_2,deuda_3,pago_3,estado_3,deuda_4,pago_4,estado_4,deuda_5,pago_5,estado_5,deuda_6,pago_6,estado_6):
    x = pd.DataFrame([[0]*len(columnas)], columns=columnas)
    x["Saldo_Limite"] = saldo_limite
    if sexo_1 == "H":
        x["Sexo"] = 1
    x["Edad"] = edad_1
    deuda = [deuda_1, deuda_2, deuda_3, deuda_4, deuda_5, deuda_6]
    deuda = [int(i) for i in deuda]
    pago = [pago_1, pago_2, pago_3, pago_4, pago_5, pago_6]
    deuda = [int(i) for i in pago]
    estado = [dic_estado[estado_1], dic_estado[estado_2], dic_estado[estado_3], dic_estado[estado_4], dic_estado[estado_5], dic_estado[estado_6]]
    for c in columnas:
        d = 0
        p = 0
        e = 0
        if "Cuenta" in c:
            x[c] = deuda[d]
            d+=1
        elif "Pago" in c:
            x[c] = pago[p]
            d+=1
        elif "Status_" in c:
            if estado[0] == int(c[-2:].replace("_","")):
                x[c] = 1
    x["Deuda"] = sum(deuda)
    if estado_civil in dic_civil.keys():
        x[dic_civil[estado_civil]] = 1
    if educacion_1 in dic_educacion.keys():
        x[dic_educacion[educacion_1]] = 1
    x['StatusFinal'] = sum(estado)
    # Check inputs are correct 
    ypred = model.predict(np.array([x.iloc[0].to_list()]))  
    return f"La probabilidad de default es de {ypred[0][0]}"

@app.callback(
    Output("graph", "figure"), 
    Input("mes-revisar", "value"))
def update_bar_chart(mes_revisar):
    cursor = engine.cursor()
    query2 = f"SELECT {dic_meses[mes_revisar]}, esdefault, count(esdefault) FROM visualizar GROUP BY {dic_meses[mes_revisar]}, esdefault;"
    df2 = pd.io.sql.read_sql_query(query2,engine)
    fig2 = px.bar(df2, x=dic_meses[mes_revisar], y='count', color="esdefault", barmode='group',
             labels={'count': 'Count', 'esdefault': 'Default'})
    fig2.update_layout(title='Count of Defaults by Education',
                    xaxis_title='Estatus',
                    yaxis_title='Count')
    return fig2

if __name__ == '__main__':
    app.run_server(debug=True)
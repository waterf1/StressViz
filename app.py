from typing import Sized
import dash
import dash_core_components as dcc
import dash_html_components as html
# from dash import dash.dcc
# from dash import html
from dash_html_components import Label
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pandas.io.formats import style
from dash.dependencies import Input, Output
import os
import io
import sklearn
from sklearn.cluster import DBSCAN
import numpy as np
import datetime as dt
import time

app = dash.Dash(__name__)
server = app.server

def cluster_DBSCAN(LOCATION):
    EPSILON_METRE = 20 #@param {type:"slider", min:25, max:500, step:25}
    MIN_POINTS = 2 #@param {type:"slider", min:5, max:30, step:1}
    R = 63710088 # Earth radius in metre

    cluster = DBSCAN(eps=EPSILON_METRE / R,
                    #The maximum angle between two samples  
                    min_samples=MIN_POINTS, 
                    #The number of samples in a neighborhood for a point to be considered as a core point.
                    metric='haversine',
                    algorithm='ball_tree' 
                    # IMPORTANT: only Ball Tree can handle haversine distance.
                    #The algorithm to compute pointwise distances and find nearest neighbors.
                    )

    loc_degrees = LOCATION.loc[:, ['latitude', 'longitude']].to_numpy() # Convert the DataFrame to a NumPy array.
    loc_radians = np.radians(loc_degrees) # To use haversine distance, degree should be transformed into radians
    labels = cluster.fit_predict(loc_radians)

    CLUSTER_DBSCAN = pd.DataFrame(
        np.column_stack([LOCATION.to_numpy(), labels]), 
        columns=['latitude', 'longitude', 'Stress', 'label']
    )
    return CLUSTER_DBSCAN

#
# showEsmRelative
# @param String {participant} ex. "1511"
# @return dataframe 
#
def showEsmRelative(participant):
    path_esm = './data/esm_data.csv'

    with open(path_esm) as f:
        head = f.readline()
        text = "\n".join([line for line in f])
        text = "\n".join([head, text])

    df_esm = pd.read_csv(io.StringIO(text),parse_dates=['responseTime_KDT'])
    df_esm = df_esm.rename(columns={'responseTime_unixtimestamp':'timestamp'})
    df_esm = df_esm.drop(['responseTime_KDT'], axis=1)
    df_esm.sort_values(by='UID', inplace=True)
    # print("df_esm.head()", df_esm.head())
    return df_esm
    

def showLocation(uid):
    df_location = readFile('LocationEntity', uid)
    df_location = df_location[['timestamp','latitude', 'longitude']]
    df_location = toDateTime(df_location)
    print("df_location\n", df_location)   
    return df_location

def resample(df, resampleString):
    df_temp = df
    print('resampled by :\n', df_temp)
    print('resampled by :', resampleString)
    df_temp.set_index('timestamp', drop=False, inplace=True)
    print('resampled by :\n', df_temp)
    df_temp = df_temp.resample(resampleString).mean()
    print('resampled by :\n', df_temp)
    df_temp = df_temp.reset_index()
    print('resampled by :\n', df_temp)
    df_temp = timeAndDate(df_temp)
    return df_temp

def timeAndDate(df):
    df['time'] = df['timestamp'].dt.time
    df['date'] = df['timestamp'].dt.date
    return df

def readFile(title, selected_UID):
    string_UID = ''
    if selected_UID < 1000:
        string_UID = '0' + str(selected_UID)
    else:
        string_UID = str(selected_UID)
    df = pd.DataFrame([])
    path = './data/P' + string_UID
    files = [f for f in os.listdir(path) if (title in f)]

    for file in files:
        df_temp = pd.read_csv(path+'/'+file)
        df = pd.concat([df, df_temp]) 

    return df   

def toDateTime(df, unit='ms'):
    df['DateTime'] = pd.to_datetime(df['timestamp'], unit=unit)
    df = df.drop('timestamp', axis=1)
    df = df.rename(columns={'DateTime':'timestamp'})

    return df

def showHearRate():
    df_heartrate = pd.DataFrame([])
    title = 'HeartRate'
    path = './P0701'
    df_heartrate = readFile('HeartRate', 701)
    # files = [f for f in os.listdir(path) if (title+'-' in f)]

    # for file in files:
    #     df = pd.read_csv(path+'/'+file)
    #     df_heartrate = pd.concat([df_heartrate, df])

    df_heartrate.drop(columns = ['Quality'], inplace=True) 
    df_heartrate.columns = ['timestamp','BPM']

    return df_heartrate
    # df_heartrate.set_index('timestamp', drop=False, inplace=True) 

#
# df_esm
#
df_esm = showEsmRelative('701')
df_esm.sort_values(by='UID', inplace=True)
df_esm = timeAndDate(toDateTime(df_esm, 's'))

#
# df_esm_personal / df_esm_relative 
#

df_esm_relative = df_esm.groupby(['UID', 'date']).mean()
df_esm_relative = df_esm_relative.reset_index('UID')
df_esm_relative['UID'] = df_esm_relative['UID'].astype('category')

#
# df_esm_median 
#
df_esm_median = df_esm_relative.reset_index('date')[['UID', 'Stress']]
df_esm_median = df_esm_median.groupby('UID').median()
df_esm_median = df_esm_median.sort_values(by=['Stress'], axis=0).reset_index("UID")
esm_uid_by_median = sorted(df_esm_median['UID'].unique())
# print(df_esm_median['UID'].unique())
# uidToMedian = zip()

#
# df_location
#
df_location = showLocation(701)
print('df_location\n', df_location.head())

#
# df_location_stress
# location X stress
#
def showLocationStress(uid):
    df_esm_personal = df_esm.loc[df_esm['UID'] == uid]
    df_esm_personal_1T = resample(df_esm_personal, '1T')

    df_location = showLocation(uid)
    df_location_1T = resample(df_location, '1T')
    
    # print('df_esm_personal_1T\n', df_esm_personal_1T.head())
    print('df_location_1T\n', df_location_1T.head())
    df_location_stress = pd.merge(df_location_1T, df_esm_personal_1T)
    df_location_stress[['latitude', 'longitude']] = df_location_stress[['latitude', 'longitude']].fillna(method = 'ffill').fillna(method = 'bfill')
    df_location_stress = df_location_stress.dropna(subset=['UID'])
    print('df_location_stress\n', df_location_stress)
    return df_location_stress


# df_location =df_location.astype({'latitude' : float, 'longitude': float})
df_location_resampled = resample(df_location, '1000L')


#
# df_heartrate
#
df_heartrate = toDateTime(showHearRate())
df_heartrate_resampled = resample(df_heartrate, '1000L')
df_heartrate_resampled = timeAndDate(df_heartrate_resampled)

heartrate = pd.DataFrame()
for i in range(1, 8):
    heartrate = pd.concat([heartrate, pd.read_csv(f"./data/P0701/HeartRate{i}.csv")])
heartrate.set_index('timestamp', drop=True, inplace=True) 
heartrate.drop(columns = ['Quality'], inplace=True) 
heartrate.columns = ['BPM']
heartrate['timestamp'] = pd.to_datetime(heartrate.index, unit='ms') 
heartrate['old_timestamp'] = heartrate.index 
heartrate.set_index('timestamp', drop=True, inplace=True)
K_EmoPhone_1000L = heartrate.resample('1000L').mean()
df = K_EmoPhone_1000L

#
# df_location X heartrate
#
# df_merged = pd.merge(df_location_resampled, df_heartrate_resampled)
# df_merged = df_merged.dropna()
# print(df_merged.head())




df_location_stress = showLocationStress(701)
# print(df_heartrate_resampled.head())
# print(df_location_resampled.head())
CLUSTER_DBSCAN = cluster_DBSCAN(df_location_stress[['latitude', 'longitude', 'Stress']])
CLUSTER_DBSCAN = CLUSTER_DBSCAN.groupby('label').mean()
CLUSTER_DBSCAN = CLUSTER_DBSCAN.reset_index()
print("CLUSTER_DBSCAN\n", CLUSTER_DBSCAN.head())

colors = {"background": "#011833", "text": "#7FDBFF"}

app.layout = html.Div(
    [
        html.H1(
            "Stress-Viz",
        ),
        html.P([
            "Welcome to Stress-Viz. This is data visualization project of CS492(F) course, 2021 Fall, KAIST.", html.Br(),
            "For students and office workers who needs easy stress management, this web application supports", html.Br(),
            "1. self-reflection and motivation for stress management", html.Br(), 
            "2. discovery of possible routine adjustment to improve stress issue", html.Br()
        ]),         
        html.P([
            "Followings are not supported in this version of prototype.", html.Br(), 
            "* Some of the visualizations are unavailable for certain user ids. If this is the case, please try another user id."
        ]),         
        html.Div(
            [
                html.Div(
                    [
                        html.Label("UID"),
                        dcc.Dropdown(
                            id="UID-dropdown",
                            options=[
                                {"label": s, "value": s} for s in df_esm.UID.unique()
                            ],
                            className="dropdown",
                        ),
                    ]
                ),
            ]
        ),
        html.H2(
            "Stress Level by Time",
        ),
        html.P(
            "What are the patterns in my stress? What are the possible effects of time and activities on stress?",
        ),
        html.Div(
            [html.Div(
                [
                    html.Label("Visible"),
                    dcc.Dropdown(
                        id="Visible-dropdown",
                        options=[
                            {"label": 'Heart BPM', "value": 'Heart BPM'},
                            {"label": 'App Usage', "value": 'App Usage'},
                            {"label": 'Physical Activity', "value": 'Physical Activity'}
                        ],
                        value=[],
                        multi=True,
                        className="dropdown2",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Date"),
                    dcc.DatePickerSingle(
                        id='Calendar',
                        min_date_allowed=dt.date(2019, 5, 1),
                        max_date_allowed=dt.date(2019, 5, 31),
                        initial_visible_month=dt.date(2019, 5, 15),
                        date=None,
                        className='calendar'
                    ),
                ]
            ),]
        ),
        html.Div(dcc.Graph(id="stress vs app usage", className="chart")),
        html.H2(
            "Relative View of Stress",
        ),
        html.P(
            "How is my stress level compared to others?",
        ),            
        html.Div(dcc.Graph(id="Stress Relative View"), className="chart"),
        html.H2(
            "Stress Level by Locations",
        ),
        html.P(
            "How is my stress level related to my environments?",
        ),        
        html.Div(dcc.Graph(id="Stress by Location"), className="chart"),    
        # html.Div(
        #     [              
        #         html.Div(
        #             [
        #                 html.Label("Date"),
        #                 dcc.Dropdown(
        #                     id="date-dropdown",
        #                     options=[
        #                         {"label": date, "value": date} for date in df_heartrate_resampled.date.unique().tolist()       
        #                         # {"label": date, "value": date} for date in ['Average'] + df_heartrate_resampled.date.unique().tolist()       
        #                         # {"label": s, "value": s} for s in df.Status.unique()
        #                     ],
        #                     className="dropdown",
        #                 ),
        #             ]
        #         ),
        #         html.Div(
        #             [
        #                 html.Label("Resample-Time (ms)"),
        #                 dcc.Dropdown(
        #                     id="resampleTime-dropdown",
        #                     options=[
        #                         {"label": time, "value": time} for time in [1000, 10000, 30000, 60000]         
        #                         # {"label": s, "value": s} for s in df.Status.unique()
        #                     ],
        #                     className="dropdown",
        #                 ),
        #             ]
        #         ),                
                # html.Div(
                #     [
                #         html.Label("Average schooling years grater than"),
                #         dcc.Dropdown(
                #             id="schooling-dropdown",
                #             options=[
                #                 # {"label": y, "value": y}
                #                 # for y in range(
                #                 #     int(df.Schooling.min()), int(df.Schooling.max()) + 1
                #                 # )
                #             ],
                #             className="dropdown",
                #         ),
                #     ]
                # ),
        #     ],
        #     className="row",
        # ),

        # html.Div(dcc.Graph(id="Stress by Heartrate"), className="chart"),
        # dcc.Slider(
        #     "resample-slider",
        #     min=1000,
        #     max=30000,
        #     step=None,
        #     marks={1000 : '1000', 5000 : '1000', 30000 : '1000', 60000 : '1000', 300000 : '1000', 600000 : '1000'},
        #     value=1000,
        # ),
    ],
    className="container",
)

@app.callback(
    Output("stress vs app usage", "figure"),
    Input("UID-dropdown", "value"),
    Input("Visible-dropdown", "value"),
    Input("Calendar", "date")
)
def update_figure(selected_UID, vis_options, date_value):
    if (selected_UID == None):
        fig_tmp = go.Figure()
        fig_tmp.update_layout(title='No Data :(', plot_bgcolor='rgba(0, 0, 0, 0.1)')
        return fig_tmp
    else:
        df_esm = pd.read_csv("./data/esm_data.csv")
        df_esm.sort_values(by='UID', inplace=True)
        df = df_esm.loc[lambda x: x['UID'] == selected_UID, :]

        df['time'] = pd.to_datetime((df['responseTime_unixtimestamp'] + 32400) * 1000, unit='ms')
        df.sort_values(by='time', inplace=True)
        df['Stress'] = df['Stress'].astype('category')

        fig_esm = go.Scatter(name = 'stress level',
                            x = df.time, y = df.Stress,
                            marker = {'color': [0 if x > 0 else 1 for x in df['Stress']], 'colorscale': [[0, 'red'], [0.5, 'blue'], [1, 'green']], 'size': 10},
                            line = {'color': 'black'},
                            mode = 'lines+markers')
        
        fig_list = []
        
        df_app = pd.DataFrame()
        string_UID = ''
        if selected_UID < 1000:
            string_UID = '0' + str(selected_UID)
        else:
            string_UID = str(selected_UID)
        
        try:
            for i in range(1, 8):
                df_app = pd.concat([df_app, pd.read_csv(f"./data/P{string_UID}/AppUsageEventEntity{i}.csv")])
        except FileNotFoundError:
            fig_tmp = go.Figure()
            fig_tmp.update_layout(title='No Data :(', plot_bgcolor='rgba(0, 0, 0, 0.1)')
            return fig_tmp
        else:
            if 'Heart BPM' in vis_options:
                df_hr = pd.DataFrame()
                
                for i in range(1, 8):
                    df_hr = pd.concat([df_hr, pd.read_csv(f"./data/P{string_UID}/HeartRate{i}.csv")])
                
                df_hr = df_hr.loc[lambda x: x['Quality'] == 'LOCKED']
                df_hr['time'] = pd.to_datetime((df_hr['timestamp'] + 32400000), unit='ms')
                df_hr.set_index('time', inplace=True)
                
                df_hr_re = pd.DataFrame()
                df_hr_re['BPM'] = df_hr.BPM.resample('1T').first()
                
                fig_hr = go.Scatter(name='BPM', x=df_hr_re.index, y=df_hr_re['BPM'], mode='lines', line_color='blue')
                fig_list.append(fig_hr)
            
            if 'App Usage' in vis_options:
                df_app = df_app.loc[lambda x: (x['type'] == 'MOVE_TO_FOREGROUND') | (x['type'] == 'MOVE_TO_BACKGROUND'), :]
                df_app['time'] = pd.to_datetime((df_app['timestamp'] + 32400000), unit='ms')

                df_msg = df_app.loc[lambda x: (x['name'] == '카카오톡') | (x['name'] == '텔레그램') | (x['name'] == 'WhatsApp'), :][['time', 'name', 'type']]
                df_msg['app_category'] = 'Messengers'
                df_msg.set_index('time', inplace=True)
                df_sns = df_app.loc[lambda x: (x['name'] == 'Facebook') | (x['name'] == '트위터') | (x['name'] == 'Instagram'), :][['time', 'name', 'type']]
                df_sns['app_category'] = 'Social Media'
                df_sns.set_index('time', inplace=True)
                df_youtube = df_app.loc[lambda x: (x['name'] == 'YouTube') | (x['name'] == '네이버TV') | (x['name'] == 'Twitch'), :][['time', 'name', 'type']]
                df_youtube['app_category'] = 'Streaming'
                df_youtube.set_index('time', inplace=True)
                df_webtoon = df_app.loc[lambda x: (x['name'] == '네이버 웹툰') | (x['name'] == '다음웹툰') | (x['name'] == '카카오페이지') | (x['name'] == '레진코믹스'), :][['time', 'name', 'type']]
                df_webtoon['app_category'] = 'Webtoon'
                df_webtoon.set_index('time', inplace=True)
                df_music = df_app.loc[lambda x: (x['name'] == 'Youtube Music') | (x['name'] == '지니 뮤직') | (x['name'] == 'VIBE'), :][['time', 'name', 'type']]
                df_music['app_category'] = 'Music'
                df_music.set_index('time', inplace=True)
                df_internet = df_app.loc[lambda x: (x['name'] == '삼성 인터넷') | (x['name'] == 'Chrome') | (x['name'] == '디시인사이드') | (x['name'] == '네이버 카페') | (x['name'] == 'NAVER'), :][['time', 'name', 'type']]
                df_internet['app_category'] = 'Internet'
                df_internet.set_index('time', inplace=True)

                df_comb = pd.concat([df_msg, df_sns, df_youtube, df_webtoon, df_music, df_internet])

                df_re = pd.DataFrame()
                df_re['name'] = df_comb.name.resample('1T').first()
                df_re['app_category'] = df_comb.app_category.resample('1T').first()
                df_re['type_first'] = df_comb.type.resample('1T').first()
                df_re['type_last'] = df_comb.type.resample('1T').last()
                df_re['using'] = (df_re['name'].notnull() * 1).astype(str)
                df_re['time'] = df_re.index

                fig_app = go.Scatter(name='app_name', x=df_re['time'], y=df_re['app_category'], mode='markers', marker={'color': 'black', 'size': 10}, hovertext=list(df_re['name']))
                fig_list.append(fig_app)

            if 'Physical Activity' in vis_options:
                df_act = pd.DataFrame()
                for i in range(1, 8):
                    df_act = pd.concat([df_act, pd.read_csv(f"./data/P{string_UID}/PhysicalActivityTransitionEntity{i}.csv")])
                df_act['time'] = pd.to_datetime((df_act['timestamp'] + 32400000), unit='ms')
                df_act.set_index('time', inplace=True)
                    
                df_act_re = pd.DataFrame()
                df_act_re['transition'] = df_act.transitionType.resample('1T').last()
                df_act_re.replace(to_replace=[None], value=np.nan, inplace=True)
                df_act_re.fillna(method='ffill', inplace=True)
                    
                def aux(s):
                    options = ['IN_VEHICLE', 'RUNNING', 'STILL', 'ON_BICYCLE', 'WALKING']
                    opt_clrs = ['blue', 'magenta', 'rgba(0, 0, 0, 0)', 'red', 'cyan']
                    for i in range(6):
                        if options[i] in s:
                            return (options[i], opt_clrs[i])
                    return ('UNKNOWN', 'black')
                        
                df_act_re['state'] = df_act_re['transition'].map(lambda x: aux(x)[0])
                df_act_re['color'] = df_act_re['transition'].map(lambda x: aux(x)[1])
                df_act_re['name'] = 'Physical State'
                
                fig_act = go.Scatter(name='state', x=df_act_re.index, y=df_act_re['name'], mode='markers', marker=dict(color = df_act_re['color'], size=10), hovertext=list(df_act_re['state']))
                fig_list.append(fig_act)
            
            l = len(fig_list)
            vis_opt_list = []
            if 'Heart BPM' in vis_options:
                vis_opt_list.append('Heart BPM')
            if 'App Usage' in vis_options:
                vis_opt_list.append('App Usage')
            if 'Physical Activity' in vis_options:
                vis_opt_list.append('Physical Activity')
            row_heights = [1] if l==0 else [0.5] + ( [(0.5)/l] * l )
            subplot_titles = ['Stress-level'] + vis_opt_list
            fig = make_subplots(rows=l+1, cols=1, row_heights=row_heights, shared_xaxes=True, subplot_titles=subplot_titles, vertical_spacing=0.05)
            fig.add_trace(fig_esm, row=1, col=1)
            for i in range(2, l+2):
                fig.add_trace(fig_list[i-2], row=i, col=1)
            fig.update_yaxes(range=[-3.5, 3.5], row=1, col=1)
            fig.update_layout(xaxis={'type': 'date'}, title_text='Stress-level vs Time', height=800, title_x = 0.5, showlegend=False, plot_bgcolor='rgba(0, 0, 0, 0.1)')
            
            if date_value is not None:
                date_L = [int(x) for x in date_value.split('-')]
                fig.update_layout(
                    xaxis=dict(
                            range=[dt.date(date_L[0], date_L[1], date_L[2]), dt.date(date_L[0], date_L[1], date_L[2])+dt.timedelta(days=1)]
                        )
                    )
            #fig.show()

            return fig

@app.callback(
    Output("Stress Relative View", "figure"),
    Input("UID-dropdown", "value"),
)
def update_esm_relative(selected_UID):

    df_esm_relative_color = df_esm_relative
    df_esm_relative_color['color'] = df_esm_relative_color["UID"] == selected_UID
    fig = go.Figure()
    # fig.add_trace(go.Box(
    #     x =
    #     y =
    #     name = 
    # ))
    fig = px.box(df_esm_relative_color, x = 'UID', y = 'Stress', color = 'color')
    fig.update_xaxes(type='category', 
    categoryorder = 'array', 
    categoryarray = esm_uid_by_median,         
    rangeslider=dict(
            visible=False
        ),)
    fig.layout.showlegend = False
    return fig

@app.callback(
    Output("Stress by Location", "figure"),
    Input("UID-dropdown", "value"),
)
def update_location(selected_UID):
    print('selected_UID', selected_UID)
    fig = go.Figure()

    if selected_UID != None:      
        MAPBOX_ACCESS_TOKEN = 'pk.eyJ1Ijoia2VsdHBvd2VyMCIsImEiOiJjazFiZ3cxZzUwMjVhM2hyMTBvcHYwcHlxIn0.mZTYvOHmJeqBANdFC1HFkw' #@param {type:"string"}


        # # CLUSTER_DBSCAN['size'] = CLUSTER_DBSCAN["Stress"]
        # CLUSTER_DBSCAN['size'] = np.exp((CLUSTER_DBSCAN["Stress"]*2))
        # for label in CLUSTER_DBSCAN.loc[:, 'label'].unique():
        #     #We will use Scattermapbox() function and defind latitude and longitude parameter
        #     sub = CLUSTER_DBSCAN.loc[lambda x: x['label'] == label, :]
            
        #     fig.add_trace(
        #         go.Scattermapbox(
        #             lat=sub.loc[:, 'latitude'],
        #             lon=sub.loc[:, 'longitude'],
        #             mode='markers',
        #             opacity=1,
        #             marker=dict(
        #                 colorscale=['#ffffb2','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#b10026'],
        #                 color = sub["Stress"],
        #                 size = sub["size"],
        #                 sizemin = 2,
        #                 # sizemode = 'diameter',
        #                 # sizeref = 2
        #             ),
        #             hovertext=sub["Stress"],
        #             hoverinfo='text'
        #         )
        #     )
        df_location_resampled = showLocationStress(selected_UID)
        df_location_resampled['size'] = np.power((df_location_resampled["Stress"] + 4)/1.5, 2)
        fig.add_trace(
            go.Scattermapbox(
                lat = df_location_resampled.loc[:, 'latitude'],
                lon = df_location_resampled.loc[:, 'longitude'],
                mode='markers',
                opacity=1,
                marker=dict(
                    # colorscale=['#ffffb2','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#b10026'],
                    cmax=3,
                    cmin=-3,
                    colorscale= "YlOrRd",
                    colorbar=dict(
                        title="Stress Level"
                    ),                
                    color=df_location_resampled["Stress"],
                    size = df_location_resampled["size"],
                    sizemin = 1,
                    # sizemode = 'diameter',
                    # sizeref = 2
                ),
                hovertext=df_location_resampled["Stress"],
                hoverinfo='text'
            )
        )

        fig.update_layout(
            autosize=True,
            mapbox=go.layout.Mapbox(
                accesstoken=MAPBOX_ACCESS_TOKEN,
                center=go.layout.mapbox.Center(
                    lat=df_location_resampled.iloc[0]['latitude'],
                    lon=df_location_resampled.iloc[0]['longitude'],
                ),
                zoom=12
            )
        )

    return fig

("""@app.callback(
    Output("Stress by Heartrate", "figure"),
    Input("resampleTime-dropdown", "value"),
    Input("date-dropdown", "value"),
    # Input("schooling-dropdown", "value"),
)
def update_heartrate(resampleInterval, selectedDate):

    resampleString = '1000L' if (resampleInterval == None) else str(resampleInterval)+'L'
    df_heartrate_resampled = resample(df_heartrate, resampleString)

    print(df_heartrate_resampled.date.astype(str).unique(), selectedDate)
    fig = px.scatter(
        df_heartrate_resampled if (selectedDate == 'Average') else df_heartrate_resampled[(df_heartrate_resampled["date"].astype(str) == selectedDate)],
        x = "time",
        y = "BPM",
        # size="",
        # color="",
        hover_name="BPM",
        size_max=60,
    )

    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font_color=colors["text"],
    )

    return fig""")

# def update_figure(selected_year, country_status, schooling):
#     filtered_dataset = df[(df.Year == selected_year)]

#     if schooling:
#         filtered_dataset = filtered_dataset[filtered_dataset.Schooling <= schooling]

#     if country_status:
#         filtered_dataset = filtered_dataset[filtered_dataset.Status == country_status]

#     fig = px.scatter(
#         filtered_dataset,
#         x="GDP",
#         y="Life expectancy",
#         size="Population",
#         color="continent",
#         hover_name="Country",
#         log_x=True,
#         size_max=60,
#     )

#     fig.update_layout(
#         plot_bgcolor=colors["background"],
#         paper_bgcolor=colors["background"],
#         font_color=colors["text"],
#     )

#     return fig


if __name__ == "__main__":
    app.run_server(debug=True)

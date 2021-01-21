import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

#How to read data from excel file
df=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\covid 19\summarised data.xlsx')
df_india=df.copy()
df

#df.drop(columns=('S.NO'),axis=1,inplace=True)
df['Total cases']=df['Confirmed']
total_cases=df['Total cases'].sum()
print(total_cases)
#df['Total cases']

df.style.background_gradient(cmap='Blues')

df['Total Active']=df['Total cases']-df['Deaths']-df['Recovered']
total_active=df['Total Active'].sum()
print(total_active)
tot_cases=df.groupby('State / UT')['Total Active'].sum().sort_values(ascending=False).to_frame()
tot_cases.style.background_gradient(cmap='Blues')


df_Full=df
f, ax=plt.subplots(figsize=(12,8))
data=df_Full[['State / UT','Total cases','Recovered','Deaths']]
data.sort_values('Total cases', ascending=False,inplace=True)
sns.set_color_codes('pastel')
sns.barplot(x='Total cases',y='State / UT',data=data,label='Total',color='r')

sns.set_color_codes('muted')
sns.barplot(x='Recovered',y='State / UT',data=data,label='Recovered',color='g')

ax.legend(ncol=2,loc="lower left", frameon=True)
ax.set(xlim=(0,45000), ylabel="States / UT",xlabel="Cases")
sns.despine(left=True,bottom=True)

dbd_India=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\CONFIRMED CASE.xlsx')
dbd_india=dbd_India.copy()
#dbd_India.tail()
#import plotly
#plotly.io.renderers.default='colab'
dbd_India.shape

import plotly
plotly.io.renderers.default = 'notebook' 
#dbd_India['Total cases']=df['Total cases']
fig=go.Figure()
fig.add_trace(go.Scatter(x=dbd_India['DATE'], y=dbd_India['Total Cases'], mode='lines+markers',name='Total Cases'))#,line=dict(shape='spline')))
fig.update_layout(title_text='Trend of coronavirus cases in India(Cumulative cases)',plot_bgcolor='rgb(230,230,230)')
fig.show()

import plotly.express as px
fig=px.bar(dbd_India, x="DATE", y="New Cases", barmode='group',height=400)
fig.update_layout(title_text='coronaovirus cases per day', plot_bgcolor='rgb(230,230,230)')
fig.show()

from fbprophet import Prophet
#df=pd.read_csv(r'C:\Users\Jatin Singh\Desktop\clean_complete.csv')
df_confirmed=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\2confirmed.xlsx')
df_death=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\2deceased.xlsx')
df_recovered=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\2recovered.xlsx')
#df_confirmed.rename(columns={'Country/Region':'Country'},inplace=True)
#df_death.rename(columns={'Country/Region':'Country'},inplace=True)
#df_recovered.rename(columns={'Country/Region':'Country'},inplace=True)
df_confirmed.head()

#grouping the dataset by date
df=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\clean_complete.xlsx')
confirmed1=df.groupby('DATE').sum()['Confirmed'].reset_index()
deaths1=df.groupby('DATE').sum()['Deaths'].reset_index()
recovered1=df.groupby('DATE').sum()['Recovered'].reset_index()
#confirmed1

fig=go.Figure()
#plotting date wise confirmed cases
fig.add_trace(go.Scatter(x=confirmed1['DATE'], y=confirmed1['Confirmed'], mode='lines+markers',name='Confirmed',line=dict(shape='spline')))
fig.add_trace(go.Scatter(x=deaths1['DATE'], y=deaths1['Deaths'], mode='lines+markers',name='Deaths',line=dict(shape='spline')))
fig.add_trace(go.Scatter(x=recovered1['DATE'], y=recovered1['Recovered'], mode='lines+markers',name='Recovered',line=dict(shape='spline')))
fig.update_layout(title='India Data of covid-19', xaxis_tickfont_size=14,yaxis=dict(title='Number of cases'))
fig.show()

from fbprophet import Prophet

#grouping the dataset by date
confirmed=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\2confirmed.xlsx')
deaths=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\2deceased.xlsx')
recovered=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\2recovered.xlsx')

#confirmed=df.groupby('Date').sum()['Confirmed'].reset_index()
#deaths=df.groupby('Date').sum()['Deaths'].reset_index()
#recovered=df.groupby('Date').sum()['Recovered'].reset_index()
#confirmed

#grouping the dataset by date
confirmed1=df.groupby('DATE').sum()['Confirmed'].reset_index()
deaths1=df.groupby('DATE').sum()['Deaths'].reset_index()
recovered1=df.groupby('DATE').sum()['Recovered'].reset_index()


confirmed1.tail()

confirmTrain=confirmed1[0:67]
confirmTest=confirmed1[68:74]
#confirmTrain
confirmTest

confirmTrain.columns=['ds','y']
confirmTrain['ds']=pd.to_datetime(confirmTrain['ds'])

#Forecasting the confirmed data using prophet
m=Prophet(interval_width=0.95)
m.fit(confirmTrain)
future=m.make_future_dataframe(periods=7)
future.tail()

#predicting the output y and the range in which output can occur
forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
confirmPredict=forecast.iloc[68:74,[0,2]]
print(confirmTest)#.iloc#[:,1]#,confirmTest.iloc[:,1])

import tensorflow as tf
x=tf.Variable(confirmPredict.iloc[:,[1]],tf.float32)
y=tf.Variable(confirmTest.iloc[1,[1]],tf.float32)
square_delta=tf.square(x-y)
loss=tf.reduce_sum(x-y)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print(sess.run(loss))

fig=go.Figure()
confirmed_forecast_plot=m.plot(forecast)
fig=confirmed_forecast_plot
#fig.add_trace(go.Scatter(x=confirmTest['DATE'], y=confirmTest['Confirmed'], mode='markers'))#,line=dict(shape='spline')))


confirmed_forecast_plot=m.plot_components(forecast)

deaths prediction

deaths1.columns=['ds','y']
deaths1['ds']=pd.to_datetime(deaths1['ds'])


deaths1.tail()

#Forecasting the confirmed data using prophet
m=Prophet(interval_width=0.95)
m.fit(deaths1)
future=m.make_future_dataframe(periods=14)
future.tail()

#predicting the output y and the range in which output can occur
forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

deaths_forecast_plot=m.plot(forecast)

deaths_forecast_plot=m.plot_components(forecast)

#Recovered

recovered1.columns=['ds','y']
recovered1['ds']=pd.to_datetime(recovered1['ds'])


recovered1.tail()

#Forecasting the confirmed data using prophet
m=Prophet(interval_width=0.95)
m.fit(recovered1)
future=m.make_future_dataframe(periods=14)
future.tail()

#predicting the output y and the range in which output can occur
forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

recovered_forecast_plot=m.plot(forecast)

recovered_forecast_plot=m.plot_components(forecast)


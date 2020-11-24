import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_excel(r'C:\Users\Jatin Singh\Desktop\confirm_india.xlsx')

def plot_series(time,series,start=0,end=None,format='-'):
    plt.plot(time[start:end],series[start:end],format)
    plt.xlabel('Time')
    plt.ylabel('Series')
    plt.grid=True

df=df.dropna()
df.shape

df.head()

series=df['Confirm']
Time=pd.to_datetime(df['Dates'])
time=np.arange(305)
plt.plot(time,series)

import plotly
import plotly.express as px
import plotly.graph_objects as go

plotly.io.renderers.default = 'notebook' 
#dbd_India['Total cases']=df['Total cases']
fig=go.Figure()
fig.add_trace(go.Scatter(x=time, y=series, mode='lines+markers',name='Total Cases'))#,line=dict(shape='spline')))
fig.update_layout(title_text='Trend of coronavirus cases in India(Cumulative cases)',plot_bgcolor='rgb(230,230,230)')
fig.show()

split_time = 300
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

batch_size=32
window_size=20
shuffle_buffer_size=300

def windowed_size(series,window_size,batch_size,shuffle_buffer):
    dataset=tf.data.Dataset.from_tensor_slices(series)
    dataset=dataset.window(window_size+1,shift=1,drop_remainder=True)
    dataset=dataset.flat_map(lambda window:window.batch(window_size+1))
    dataset=dataset.shuffle(shuffle_buffer).map(lambda window:(window[:-1],window[-1:]))
    dataset=dataset.batch(batch_size).prefetch(1)
    return dataset

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

train_set=windowed_size(series,window_size,batch_size=128,shuffle_buffer=shuffle_buffer_size)

model=tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[None]),
    tf.keras.layers.SimpleRNN(60,return_sequences=True),
    tf.keras.layers.SimpleRNN(60,return_sequences=True),
    tf.keras.layers.SimpleRNN(60,return_sequences=True),
    tf.keras.layers.SimpleRNN(60),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x:x*100.0)
])

lr_schedule=tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-8 * 10 **(epoch/20))
optimizers=tf.keras.optimizers.SGD(lr=1e-8,momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),optimizer=optimizers,metrics=["mae"])
history=model.fit(train_set,epochs=300,callbacks=[lr_schedule])

plt.semilogx(history.history['lr'],history.history['loss'])
#plt.axis([1e-8,1e-4,0,30])

epochs=range(len(history.history['loss']))
plt.plot(epochs,history.history['loss'])

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

dataset=windowed_size(series,window_size,batch_size,shuffle_buffer_size)

model=tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[None]),
    tf.keras.layers.SimpleRNN(60,return_sequences=True),
    tf.keras.layers.SimpleRNN(60,return_sequences=True),
    tf.keras.layers.SimpleRNN(60,return_sequences=True),
    tf.keras.layers.SimpleRNN(60),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x:x*100.0)
])

optimizer=tf.keras.optimizers.SGD(lr=0.1,momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),optimizer=optimizer,metrics='mae')
history=model.fit(dataset,epochs=400)

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1]



plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)

tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()


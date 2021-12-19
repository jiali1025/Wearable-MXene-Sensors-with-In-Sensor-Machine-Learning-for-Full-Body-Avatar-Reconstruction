# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/drive/1tICjIjYYjYtJbBGwwiKAIg0eSf15rODQ
"""


import os
import pathlib
from matplotlib import animation

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io as sio
from matplotlib import pyplot 
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from keras.models import load_model
from tensorflow.keras import models
from tensorflow import keras
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython import display
from sklearn.metrics import mean_squared_error
from keras import regularizers
from bayes_opt import BayesianOptimization
import time
from scipy import stats
MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_TF = MODELS_DIR + 'model'
MODEL_NO_QUANT_TFLITE = MODELS_DIR + 'model_no_quant.tflite'
MODEL_TFLITE = MODELS_DIR + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'
'''
The followings are helper functions to preprocess the data as well as
smoothing the prediction output
'''
def get_history(neural_data,bins_before,bins_after,bins_current=1):

    num_examples=neural_data.shape[0] 
    num_neurons=neural_data.shape[1] 
    surrounding_bins=bins_before+bins_after+bins_current 
    X=np.empty([num_examples,surrounding_bins,num_neurons])
    X[:] = np.NaN
    start_idx=0
    for i in range(num_examples-bins_before-bins_after):
        end_idx=start_idx+surrounding_bins; 
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] 
        start_idx=start_idx+1;
    return X
def dataset_history_processing(X_train,X_val,X_test,Y_train,Y_val,Y_test,n_hours,n_future):
  if n_hours>0:
    X_train = get_history(X_train,n_hours,n_future,1)
    X_val = get_history(X_val,n_hours,n_future,1)
    X_test = get_history(X_test,n_hours,n_future,1)

    X_train = X_train[n_hours:len(X_train)-n_future,:,:]
    X_val = X_val[n_hours:len(X_val)-n_future,:,:]
    X_test = X_test[n_hours:len(X_test)-n_future,:,:]

    Y_train = Y_train[n_hours:len(Y_train)-n_future,:]
    Y_test = Y_test[n_hours:len(Y_test)-n_future,:]
    Y_val = Y_val[n_hours:len(Y_val)-n_future,:]

  return X_train,Y_train,X_val,Y_val,X_test,Y_test

def moving_average(input_pred,N):
    [a,b]=input_pred.shape
    out_pred=np.zeros([a-N,b])
    for i in range(0,a-N):
        out_pred[i,:]=np.mean(input_pred[i:i+N,:],axis=0)
    return out_pred

'''
The followings are the different model structures, we have mainly used two different models
The CNN model and RNN model. However, this script is only for the training of the CNN model

'''

def CNN_model(filters1,filters2,LR,reg,n_hours,train_x,train_y,val_x,val_y,epochs,verbose):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=filters1,kernel_size=[1,4],activation='tanh',input_shape=(n_hours+1,7,1)))#,kernel_regularizer=regularizers.l2(0.0018)))
    model.add(layers.Conv2D(filters=filters2,kernel_size=[1,1],activation='tanh'))#,kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Flatten())
    model.add(layers.Dense(15, activation='linear'))  
    opt = keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt,  loss="mse", metrics=["mae"])
    EPOCHS =10000
    callbacks = [EarlyStopping(monitor='val_loss', patience=epochs), ModelCheckpoint(filepath='./best_model_CNN.h5', monitor='val_loss', save_best_only=True)]# uses validation set to stop training when it start overfitting
    model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=callbacks,epochs=EPOCHS,batch_size=16, verbose=verbose, shuffle=True)
    model.save('./best_model_CNN.h5')

def RNN_model(filters1,filters2,LR,reg,n_hours,train_x,train_y,val_x,val_y,epochs,verbose):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=filters1,recurrent_activation='tanh',return_sequences=True,input_shape=(7,n_hours+1),use_bias=True,kernel_initializer='random_uniform',bias_initializer='zeros',dropout=reg))#,kernel_regularizer=regularizers.l2(reg),bias_regularizer=regularizers.l2(reg),recurrent_regularizer=regularizers.l2(reg)))#,kernel_regularizer=regularizers.l2(0.0018)))
    
    model.add(layers.LSTM(units=filters2,recurrent_activation='tanh',use_bias=True,kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.add(layers.Dense(15, activation='linear'))  
    opt = keras.optimizers.RMSprop(learning_rate=LR)
    model.compile(optimizer=opt,  loss="mse", metrics=["mae"])
    EPOCHS =10000
    callbacks = [EarlyStopping(monitor='val_loss', patience=epochs), ModelCheckpoint(filepath='./best_model.h5', monitor='val_loss', save_best_only=True)]# uses validation set to stop training when it start overfitting
    model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=callbacks,epochs=EPOCHS,batch_size=16, verbose=verbose, shuffle=True)
    model.save('./best_model.h5')
    

def cnn_evaluate(filters1,filters2,LR,reg,n_hours,epochs):
  filters1=int(filters1)
  filters2=int(filters2)
  LR=float(LR)
  reg=float(reg)
  n_hours=int(n_hours)
  epochs=int(epochs)
  x_train,y_train,x_validate,y_validate,x_test,y_test=dataset_history_processing(x_train1,x_validate1,x_test1,y_train1,y_validate1,y_test1,n_hours,n_future)
  x_train=np.reshape(x_train,[-1,n_hours+1,7,1])
  x_test=np.reshape(x_test,[-1,n_hours+1,7,1])
  x_validate=np.reshape(x_validate,[-1,n_hours+1,7,1])
  CNN_model(filters1,filters2,LR,reg,n_hours,x_train,y_train,x_validate,y_validate,epochs,0)#note this is using the CNN model
  model = load_model('./best_model_CNN.h5')
  yhat_validate = model.predict(x_validate,batch_size=1)
  R=np.zeros([1,15])
  for i in range(0,15):
      R[0,i],_=stats.pearsonr(yhat_validate[:,i],y_validate[0:len(yhat_validate),i])
  return np.mean(R) #This is just done to minimized meansqure error
'''
This is the code for drawing the 2D pose of human
'''

connectivity_dict = [[0, 1, 0], [1, 2, 0], [2, 3, 0],
                              [3, 4, 0], [1, 5, 1], [5, 6, 1],
                              [6, 7, 1], [1, 8, 0], [8, 9, 0], [9, 10, 0], [10, 11, 0], [8, 12, 1], [12, 13, 1], [13, 14, 1]]



def draw2Dpose(pose_2d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):  # blue, orange
    for i in connectivity_dict:
        x, y = [np.array([pose_2d[i[0], j], pose_2d[i[1], j]]) for j in range(2)]
        ax.plot(x, y, lw=2, c=lcolor if i[2] else rcolor)

"""#Read training, validation and test data"""

data = pd.read_csv("/content/drive/MyDrive/Dataset/Tactile_Sensor/Pose_prediction/train_data.csv")
data=data.to_numpy()

x = data[:,0:7]
y = data[:,7:]
x_train1, x_validate1, y_train1, y_validate1 = train_test_split(x, y, test_size = 0.20,shuffle=False)

sc = StandardScaler ()
label_sc = StandardScaler()
sc.fit(x_train1)
x_train1=sc.transform(x_train1)
x_validate1=sc.transform(x_validate1)

label_sc.fit(y_train1)
y_train1=label_sc.transform(y_train1)
y_validate1=label_sc.transform(y_validate1)

test_data = pd.read_csv("/content/drive/MyDrive/Dataset/Tactile_Sensor/Pose_prediction/test_data.csv")
#test_data = pd.read_csv("./Training_data.csv")
test_data=test_data.to_numpy()
x_test1 = test_data[:,0:7]
y_test1 = test_data[:,7:]
x_test1=sc.transform(x_test1)
y_test1=label_sc.transform(y_test1)

n_future=0

"""#Create a model and tune the hyperparameters. Please skip  if you dont want to tune the hyperparameters"""

rnnBO = BayesianOptimization(cnn_evaluate, {'filters1': (1,500),'filters2': (1,100),'LR': (0.00001,0.01),'reg': (0.1,0.5), 'n_hours': (3,20), 'epochs': (10,10)},verbose=2)
rnnBO.maximize(init_points=20,n_iter=10,acq='poi',kappa=1,xi=1e-1,alpha=1e-6)#exploration
rnnBO.maximize(init_points=0,n_iter=10,acq='poi',kappa=1,xi=1e-4,alpha=1e-6)#exploitation

best_params=rnnBO.max['params']
filters1=np.int(best_params['filters1'])
filters2=np.int(best_params['filters2'])
LR=np.float(best_params['LR'])
n_hours=np.int(best_params['n_hours'])
epochs=np.int(best_params['epochs'])
reg=np.float(best_params['reg'])

best_params=rnnBO.max['params']
filters1=np.int(best_params['filters1'])
filters2=np.int(best_params['filters2'])
LR=np.float(best_params['LR'])
n_hours=np.int(best_params['n_hours'])
epochs=np.int(best_params['epochs'])
reg=np.float(best_params['reg'])
print(filters1)
print(filters2)
print(LR)
print(n_hours)
print(epochs)
print(reg)

filters1=313
filters2=28
LR=1e-05
n_hours=20
epochs=10
reg=0.5

x_train,y_train,x_validate,y_validate,x_test,y_test=dataset_history_processing(x_train1,x_validate1,x_test1,y_train1,y_validate1,y_test1,n_hours,n_future)

x_train=np.reshape(x_train,[-1,n_hours+1,7,1])
x_test=np.reshape(x_test,[-1,n_hours+1,7,1])
x_validate=np.reshape(x_validate,[-1,n_hours+1,7,1])

CNN_model(filters1,filters2,LR,reg,n_hours,x_train,y_train,x_validate,y_validate,epochs,2)
model = load_model('./best_model_CNN.h5')

model.summary()

"""Fit the model"""

model.save('/content/drive/MyDrive/Dataset/Tactile_Sensor/best_model.h5')
model.save(MODEL_TF)
test_loss, test_mae = model.evaluate(x_test, y_test)
y_pred_model=model.predict(x_test,batch_size=1)

y_pred=label_sc.inverse_transform(y_pred_model)#This is what would plotted for GIF
y_test_orig=label_sc.inverse_transform(y_test)

pyplot.plot(y_pred_model[530,:])
pyplot.plot(y_test[530,:])
print(y_pred[530,:] - y_test_orig[530,:])

"""Plot the gif"""

sample=500

x_axis=np.array([792.223,809.895,709.828,692.074,671.686,907.068,954.199,968.828,824.605,762.728,700.014,742.115,886.383	,970.571,951.229])
y_pred_avg=moving_average(y_pred,4) #You can change this from 4 to smaller number 
y_test_tflite = {"x_test":x_test,"y_test_orig":y_test_orig,"y_pred_avg":y_pred_avg}
sio.savemat('./tflite_pred.mat',y_test_tflite)
fig = plt.figure(figsize=(4,8))
ax = fig.add_subplot()
plt.xlim([500, 1170])
plt.ylim([0, 1100])
ax.invert_xaxis()
ax.invert_yaxis()
draw2Dpose(np.array([x_axis,y_pred[sample,:]]).T, ax)
for i in range(500,650):
  fig = plt.figure(figsize=(4,8))
  ax = fig.add_subplot()
  plt.xlim([500, 1170])
  plt.ylim([0, 1100])
  ax.invert_xaxis()
  ax.invert_yaxis()
  draw2Dpose(np.array([x_axis,y_pred_avg[i,:]]).T, ax)

R=np.zeros([1,15])
for i in range(0,15):
    R[0,i],_=stats.pearsonr(y_pred_avg[:,i],y_test[0:len(y_pred_avg),i])
print("Pearson R")
print(np.mean(R))

sMAPE = 0
for i in range(y_test_orig.shape[0]):
        sMAPE += np.mean(abs(y_pred[i,:] - y_test_orig[i,:]) / (y_pred[i,:] + y_test_orig[i,:]) / 2) / (y_test_orig.shape[0])
print("sMAPE: {}%".format(sMAPE * 100))



print("MEAN squared ERROR")
print(mean_squared_error(label_sc.transform(y_pred_avg),y_test[0:len(y_pred_avg),:]))

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
model_no_quant_tflite = converter.convert()
# Save the model to disk
open(MODEL_NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)
def representative_dataset():
  for i in range(50):
    yield([x_train[i,:,:,0].reshape(1,n_hours+1,7,1).astype(np.float32)])

# Set the optimization flag.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce integer only quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

# Save the model to disk
open(MODEL_TFLITE, "wb").write(model_tflite)

def predict_tflite(tflite_model, x_test):
  # Prepare the test data
  x_test_ = x_test.copy()
  x_test_ = x_test_.reshape((1,n_hours+1,7,1))
  x_test_ = x_test_.astype(np.float32)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # If required, quantize the input layer (from float to integer)
  input_scale, input_zero_point = input_details["quantization"]
  if (input_scale, input_zero_point) != (0.0, 0):
    x_test_ = x_test_ / input_scale + input_zero_point
    x_test_ = x_test_.astype(input_details["dtype"])
  # Invoke the interpreter

  interpreter.set_tensor(input_details["index"], x_test_)
  interpreter.invoke()
  y_pred = interpreter.get_tensor(output_details["index"])[0]
  
  # If required, dequantized the output layer (from integer to float)
  output_scale, output_zero_point = output_details["quantization"]
  if (output_scale, output_zero_point) != (0.0, 0):
    y_pred = y_pred.astype(np.float32)
    y_pred = (y_pred - output_zero_point) * output_scale

  return y_pred

#predict_tflite(model_no_quant_tflite, x_test[1,:,:,:])
print(y_pred.shape)

"""Note depending on the model the quantized version might have higher accuracy. """

# Calculate predictions with full software
y_pred = label_sc.inverse_transform(model.predict(x_test))
print("Mean squared Error for Full model")
print(mean_squared_error(y_test_orig,y_pred))
y_test_pred_no_quant_tflite=np.empty([x_test.shape[0],15])
y_test_pred_tflite=np.empty([x_test.shape[0],15])
# Calculate predictions with tensorflow lite
for i in range(0,x_test.shape[0]):
  y_test_pred_no_quant_tflite[i,:]=(predict_tflite(model_no_quant_tflite, x_test[i,:,:,:]))
y_test_pred_no_quant_tflite=label_sc.inverse_transform(y_test_pred_no_quant_tflite)
print('Test accuracy with model tf lite:')
print(mean_squared_error(y_test_orig,y_test_pred_no_quant_tflite))

# Calculate predictions with tensorflow lite quantized model
for i in range(0,x_test.shape[0]):
  y_test_pred_tflite[i,:]=(predict_tflite(model_tflite, x_test[i,:,:,:]))
y_test_pred_tflite=label_sc.inverse_transform(y_test_pred_tflite)

print('Test accuracy with model quantized:')
print(mean_squared_error(y_test_orig,y_test_pred_tflite))


y_test_tflite = {"y_test_pred_tflite":y_test_pred_tflite,"x_test":x_test,"y_test_orig":y_test_orig}
sio.savemat('models/tflite_pred.mat',y_test_tflite)

# Calculate size
size_no_quant_tflite = os.path.getsize(MODEL_NO_QUANT_TFLITE)
size_tflite = os.path.getsize(MODEL_TFLITE)

# Compare size
# Compare size
pd.DataFrame.from_records(
    [["TensorFlow Lite", f"{size_no_quant_tflite} bytes ", f"(reduced by {0} bytes)"],
     ["TensorFlow Lite Quantized", f"{size_tflite} bytes", f"(reduced by {size_no_quant_tflite - size_tflite} bytes)"]],
     columns = ["Model", "Size", ""], index="Model")

# # Install xxd if it is not available
# !apt-get update && apt-get -qq install xxd
# # Convert to a C source file, i.e, a TensorFlow Lite for Microcontrollers model
# !xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}
# # Update variable names
# REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
# !sed -i 's/'{REPLACE_TEXT}'/g_model/g' {MODEL_TFLITE_MICRO}
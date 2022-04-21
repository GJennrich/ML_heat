#!/usr/bin/python
"""
Script for Machine Learning approach to week 2 Extreme Heat
Using GEFSv12 Tmax to give a more skillful Tmax forecast
Target= Week 2 Tmax (any day +8 to +14- day independent)
Future Target= Week 2 Tmax (dependecy on forecast day)(each forecast day has different weights)
original author: @Greg Jennrich
"""
#ML inputs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Input, concatenate
from tensorflow.keras.models import Sequential, save_model, load_model, Model
import tensorflow.keras.backend as K
#other inputs
import sys,io,os
import netCDF4
from contextlib import redirect_stdout
import numpy as np
from tqdm import tqdm
import calendar
from datetime import date,timedelta,datetime
import xarray as xr  # for regrid
import xesmf as xe
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
#use Li's functions
import xu
from xu import ndates,Ds,mmean,rmse,bias,cf,atxt
from xu import rmse as RMSE
#########################################################################################
project_dir='/cpc/home/gjennrich/'
#######
#read in CONUS/AK domain
def get_conus_mask(degree):
	conus_file='/cpc/home/gjennrich/Scripts/ML_heat/conus_grid_CDAS_heat.nc'
	US_mask = xr.open_dataset(conus_file, engine='netcdf4')['mask_array']\
				.interp(lat=np.arange(-90, 90.5,degree), lon=np.arange(0, 360,degree),method='nearest')
	US_mask= US_mask.sel(lat=slice(20,50)).sel(lon=slice(231,301)).transpose('lon','lat')
	return US_mask
def get_target_mask(degree):
	conus_file='/cpc/home/gjennrich/Scripts/ML_heat/full_mask.nc'
	US_mask = xr.open_dataset(conus_file, engine='netcdf4')['land_mask']\
				.interp(lat=np.arange(-90, 90.5,degree), lon=np.arange(0, 360,degree),method='nearest')
	US_mask= US_mask.sel(lat=slice(20,50)).sel(lon=slice(231,301)).transpose('lon','lat')
	return US_mask
def Get_dateime_date(var_time):#time coord
	time_coords=[]
	for t in range(var_time.shape[0]):
		#current convenstion is 'days since Jan 0 0000'
		new_date=date(1,1,1)+timedelta(int(var_time[t].values))-timedelta(367)#365+1leapday+day 0 start
		time_coords.append(new_date)
	var_time['time']=(np.asarray(time_coords)).astype('datetime64')
	return var_time['time']
def Get_forecast_dates(forecast_time_xr,dayofarray=-1):#for a given initalization date, get the forcast dates
	int_date=pd.DatetimeIndex(forecast_time_xr.values)[dayofarray]
	forecast_dates=pd.date_range(int_date+timedelta(8),int_date+timedelta(14))#Day +8 to +14
	return forecast_dates
def M2V(D):#Li's code
	'''convert from 2D Matrix to 1D vector, eliminate all undefined ocean
	[...,71,31] to [...,865]'''
	#2201 total, 865 good points (CONUS)
	#mask=np.load('mask.npy').transpose()
	mask=get_target_mask(1).values.astype('bool')
	return D[...,mask]
def Remove_nans_1D(map_1d):
	''' eliminate all undefined ocean poi'''
	#2201 total, 865 good points (CONUS)
	mask=get_target_mask(1).astype('bool')
	mask_flat=mask.stack(point=('lon','lat'))
	map_cut=map_1d.sel(point=mask_flat)
	return map_cut
def Add_nan_mask(map_1d):
	mask=get_target_mask(1).astype('bool')
	mask_flat=mask.stack(point=('lon','lat'))
#
"""
Read in Evan's GEFSv12 data and his obs data For Tmax
"""
#fcst1='/cpc/gth/GTH_DATABASE/WEEK2_HEAT/GEFS_SEHOS/GEFS_v12_reforecast_airT.nc'
fcst1='/cpc/gth/GTH_DATABASE/WEEK2_HEAT/GEFS_SEHOS/GEFS_v12_reforecast_airT.nc'
t_max_f=Ds(fcst1)['max_air_temp_2m'];intal_date=Ds(fcst1)['issue_date']
#obs1='/cpc/home/evan.oswald/R1_SEHOS/CDAS_1x1_reanalysis_airT.nc'
obs1='/cpc/home/evan.oswald/R1_SEHOS/CDAS_1x1_reanalysis_airT.nc'
obs=Ds(obs1)['max_air_temp_2m'];obs_date=Ds(obs1)['issue_date']

#Get datetime objects for time
t_max_f['time']=Get_dateime_date(intal_date)
obs['time']=Get_dateime_date(obs_date)
#Get dates for the forecast
t_max_f['fcst_date']=np.arange(8,15);t_max_f['ensemble']=np.arange(1,6)#give the day+ forecast number
t_max_f['longitude']=np.arange(231,302);obs=obs[:,::-1,:]#get the same lon and lats!

#test
Get_forecast_dates(t_max_f['time'],0)

"""
Cut down to 2000-2020, to remove 'bad' gefs data
Remove missing Obseravtions (June and July 2015)
"""
t_max_f=t_max_f.sel(time=slice(date(2000,1,1),date(2020,12,31)))

missing_idx=np.where(np.isnan(obs[2,2,:].values))[0]#known good point
missing_dates=pd.DatetimeIndex(obs.time.isel(time=missing_idx).values)
obs=obs.isel(time=np.delete(np.arange(0,obs.time.values.shape[0]),missing_idx))#remove missing dates

"""
Now take the ens mean (just to be simple) and connect forecast dates with obs dates
24255= 3465 int dates * 7 forecast dates
"""
ens_mean=False
if ens_mean:
	t_max_ens_mean=t_max_f.mean(dim='ensemble')
	inputs=t_max_ens_mean.stack(fcst=('time','fcst_date'))
else:
	inputs=t_max_f.stack(fcst=('time','fcst_date','ensemble'))
input_dates=[]
for d in tqdm(range(inputs.shape[-1])):
	#date_add=pd.DatetimeIndex(inputs.time.values)[d]+timedelta(int(inputs.fcst_date.values[d]))
	date_add=pd.DatetimeIndex(inputs.fcst.time.values)[d]
	input_dates.append(date_add)

input_dates_full=pd.DatetimeIndex(np.asarray(input_dates))
input_dates_cut=input_dates_full[~np.in1d(input_dates_full,missing_dates)]#remove missing obs days
inputs=inputs[:,:,~np.in1d(input_dates_full,missing_dates)]#remove missing obs days
inputs['fcst']=input_dates_cut;inputs=inputs.transpose('fcst','longitude','latitude')

target=obs.sel(time=input_dates_cut).transpose('time','longitude','latitude')#obs dates now aline with forecast dates :)

"""
split training and testing
80/20 split, shuffle
scale
"""
#vectorize first and remove nans!
#inputs_flat=M2V(inputs.values);target_flat=M2V(target.values)
inputs_flat=inputs.stack(point=('longitude','latitude'));target_flat=target.stack(point=('longitude','latitude'))
inputs_flat_cut=Remove_nans_1D(inputs_flat);target_flat_cut=Remove_nans_1D(target_flat)
#inputs_flat=xr.where(np.isnan(inputs_flat),0,inputs_flat);target_flat=xr.where(np.isnan(target_flat),0,target_flat)
scaler_input= MinMaxScaler(feature_range=(-1,1));scaler_target= MinMaxScaler(feature_range=(-1,1))
#inputs_flat_scaled=scaler_input.fit_transform(inputs_flat_cut);target_flat_scaled=scaler_target.fit_transform(target_flat_cut)
input_train_, input_test_, target_train_, target_test_= train_test_split(inputs_flat_cut,target_flat_cut,test_size=0.2,shuffle=True)
input_train=scaler_input.fit_transform(input_train_);input_test=scaler_input.fit_transform(input_test_)
target_train=scaler_target.fit_transform(target_train_);target_test=scaler_target.fit_transform(target_test_)


"""
Deterministic ML Model
Input: GEFSv12 Tmax Ens mean forecasts
Target: Observed Tmax
Samples:  2000-2020 Daily GEFSv12 forecasts Day +8 to +14 (each as indivdual sample)
Note: We assume here that +8 and +14 have the same biases, no difference in weights
"""
#IF we want unscaled RMSE we can use this but can just use the unscaled one
def rmse_unscaled_metric(y_true, y_pred):
	y_true2=scaler_input.inverse_transform(y_true)
	y_pred2=scaler_target.inverse_transform(y_pred)
	E = y_pred2 - y_true2
	return np.sqrt(np.mean(E * E))
def rmse_metric(y_true, y_pred):
	E = y_pred - y_true
	return K.sqrt(K.mean(E * E))

batch_size = 256
layers = 3 #experiment, start with 2, work upward
nodes = 4 #, work up in increments of 10
epochs = 50 #50-ens mean #10-with ens
activation_function =tf.nn.tanh
from tensorflow.keras.models import Model
input1 = Input(shape=(1306,)) #GEFS temp
output1 = Dense(1306, activation="tanh")(input1)
output2 = Dense(1306, activation="tanh")(output1)
#output2 = Dense(1306, activation="tanh")(output1)
Model = Model(inputs=input1, outputs=output2)
Model.summary()#print a summary of the model
#compile
optimizer = 'adam'#keras.optimizers.RMSprop(0.0001)
Model.compile(loss='mse',optimizer=optimizer,metrics=['mae',rmse_metric])#run_eagerly=True
t_board = [keras.callbacks.TensorBoard(log_dir='/cpc/home/gjennrich/my_tf_dir/Tmax_ens_member_2layer',histogram_freq=1,embeddings_freq=1,)]
training_output=Model.fit(x=input_train,y=target_train,
				validation_data=(input_test,target_test),
				batch_size=batch_size,epochs=epochs, shuffle=False, verbose=1,callbacks=[t_board])
# #evaluate the testing (validation) data
# evaluate_output=Model.evaluate(x=input_test,y=target_test,batch_size=batch_size)
# #evaluate with test data
# print(evaluate_output)

#test model output
nn_prediction=Model.predict(input_test)
nn_map_1d=scaler_target.inverse_transform(nn_prediction)
#turn back into an xarray, and unstack to get 2d map (57*25)
nn_map=target_flat[0:nn_map_1d.shape[0],:].copy()*0
mask=get_target_mask(1).astype('bool').stack(point=('lon','lat'))
nn_map[...,mask]=nn_map_1d;nn_map=nn_map.unstack('point')
#input unstack for plotting
input_2d=target_flat[0:input_test_.shape[0],:].copy()*0
input_2d[...,mask]=input_test_.values;input_2d=input_2d.unstack('point')
#target unstack for plotting
target_2d=target_flat[0:target_test_.shape[0],:].copy()*0
target_2d[...,mask]=target_test_.values;target_2d=target_2d.unstack('point')
nn_map['time']=target_test_['time'];input_2d['time']=target_test_['time'];target_2d['time']=target_test_['time']
#Score rmse
rmse=[];rmse_GEFS=[]
for d in range(nn_map_1d.shape[0]):
	rmse_day= sqrt(mean_squared_error(nn_map_1d[d,:],target_test_.isel(time=d).values))
	rmse.append(rmse_day)
	rmse_day= sqrt(mean_squared_error(input_test_.isel(fcst=d),target_test_.isel(time=d).values))
	rmse_GEFS.append(rmse_day)
print('Mean ML RMSE: ' +str(np.asarray(rmse).mean()))
print('Mean GEFSv12 RMSE: ' +str(np.asarray(rmse_GEFS).mean()))

#RMSE Maps
gefs_maps=RMSE(input_2d.values,target_2d.values,axis=0)
NN_maps=RMSE(nn_map.values,target_2d.values,axis=0)

# Model=Sequential(name='Tmax')
# Model.add(Dense(nodes, activation=activation_function,input_shape=(5,)))
# Dropout(0.3)
# for l in range(layers-2):
# 	Model.add(Dense(nodes, activation=activation_function))
# 	Dropout(0.3)
# Model.add(Dense(5, activation=activation_function))
# Dropout(0.3)
# Model.summary()#print a summary of the model
# #compile
#
# Model.compile(loss='mse',optimizer='adam',metrics='mae')
# #train the model
# input_train_tf=tf.convert_to_tensor(input_train,np.float32)
# target_train_tf=tf.convert_to_tensor(target_train,np.float32)
# training_output=Model.fit(input_train_tf[0:25,0:5],target_train_tf[0:25,0:5],batch_size=batch_size,epochs=epochs, shuffle=True, verbose=1)
#
# #evaluate with test data
# evaluate_output=Model.evaluate(input_test[0:25,0:5],target_test[0:25,0:5],batch_size=batch_size)
# print(evaluate_output)
#
# #test model output
# test_prediction=Model.predict([input_test[0:25,0:5]])









"""
PLOTS:

Plot sample forecast:
-ML model output
-GEFS raw output
-Observations
"""
import matplotlib.pyplot as plt
import matplotlib.colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

crs = ccrs.PlateCarree(central_longitude=360)
cmap = plt.cm.YlOrRd
def plot_background(ax1):
	#ax1.set_extent([-35, -100, 25.0, 53], crs=ccrs.PlateCarree())
	ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
	ax1.yaxis.set_major_formatter(LatitudeFormatter())
	ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
	ax1.add_feature(cfeature.STATES, linewidth=0.5)
	ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
	ax1.gridlines(crs, linewidth=0.5, color='lightgray', linestyle='--')

fig, axarr = plt.subplots(3, 1, figsize=(14, 8), subplot_kw={'projection': crs}) #make the map bigger than the table
#hemispheric view
ax1=axarr[0];ax2=axarr[1];ax3=axarr[2];

plot_background(ax1);plot_background(ax2);plot_background(ax3)
pick_date=-50#date(2011,7,29)
#print_skill=[hss_NN[pick_date],hss_model_ens[pick_date],hss_LR[pick_date]]
ax1.set_title('GEFSv12 Tmax Forecast RMSE: '+str(round(rmse_GEFS[pick_date],2)))
cf1 = ax1.contourf(input_2d.longitude, input_2d.latitude, input_2d.isel(time=pick_date).transpose(), 17,cmap=cmap,levels=np.linspace(60,120,17),vmax=120, vmin=60,extend='both')
ax2.set_title('NN Corrected Tmax Forecast RMSE: '+str(round(rmse[pick_date],2)))
cf2 = ax2.contourf(nn_map.longitude,nn_map.latitude,  nn_map.isel(time=pick_date).transpose(),17,cmap=cmap,levels=np.linspace(60,120,17),vmax=120, vmin=60,extend='upper')
ax3.set_title('Observation: '+pd.DatetimeIndex(target_2d.time.values)[pick_date].strftime('%b %d %Y'))
cf3 = ax3.contourf(target.longitude, target.latitude,  target_2d.isel(time=pick_date).transpose(), 17,cmap=cmap,levels=np.linspace(60,120,17),vmax=120, vmin=60,extend='both')
#cbar = fig.colorbar(cf1, orientation='horizontal', ax=ax2,extend='both')# shrink=.5)
#cbar.outline.set_edgecolor('black');cbar.outline.set_linewidth(.5)
#cbar.ax.set_xlabel('Anomaly (C)', fontsize='x-large')
#plt.legend()
fig.suptitle('Week 2 Tmax Forecast', fontsize='18')
#plt.savefig(project_dir+'model_NN_NN2_obs_prob_ex.png')
plt.show()

"""
RMSE Map
"""
fig, axarr = plt.subplots(2, 1, figsize=(14, 8), subplot_kw={'projection': crs}) #make the map bigger than the table
#hemispheric view
ax1=axarr[0];ax2=axarr[1];

plot_background(ax1);plot_background(ax2)
ax1.set_title('GEFSv12 HI Forecast RMSE: '+str(round(np.nanmean(rmse_GEFS),2)),fontsize='18')
cf1 = ax1.contourf(input_2d.longitude, input_2d.latitude, gefs_maps.transpose(), 17,cmap=cmap,levels=np.linspace(0,15,11),vmax=15, vmin=0,extend='max')
ax2.set_title('NN Corrected HI Forecast RMSE: '+str(round(np.nanmean(rmse),2)),fontsize='18')
cf2 = ax2.contourf(nn_map.longitude,nn_map.latitude, NN_maps.transpose(),17,cmap=cmap,levels=np.linspace(0,15,11),vmax=15, vmin=0,extend='max')
cax = fig.add_axes([0.8, 0.25, 0.04, 0.5])
cbar = fig.colorbar(cf1, cax=cax,extend='max')# shrink=.5)
cbar.outline.set_edgecolor('black');cbar.outline.set_linewidth(.5)
cbar.ax.set_ylabel('RMSE', fontsize='x-large')
#plt.legend()
fig.suptitle('Week 2 Heat Index Forecast\n(Any Day, any Ens Member)', fontsize='20')
plt.savefig('heat_index_rmse_map_ens_members.png')
plt.show()


"""
Plot line graph
"""
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import MonthLocator, DateFormatter
#temp
fig, ax=plt.subplots(figsize=(18,6))
plt.scatter((nn_map.time).sort(), rmse, color='blue',label='ML RMSE ('+"{:.1f}".format(np.nanmean(rmse))+')')
plt.scatter((nn_map.time), rmse_GEFS, color='green',label='GEFS RMSE ('+"{:.1f}".format(np.nanmean(rmse_GEFS))+')')
plt.title('Temperature: Week 3/4 2-Cat HSS', fontsize=22,fontweight='bold', loc='left')
plt.xlabel('Forecast Date', fontsize=18,fontweight='bold')
plt.ylabel('Root Mean Squared Error', fontsize=18,fontweight='bold')
ax.set_yticks(np.arange(0,20,20))
plt.grid(True,linewidth=1)
fig.legend(ncol=5,fontsize='x-large')
fig.tight_layout()
#plt.savefig('tmax_rmse_line.png',dpi =150)
plt.show()


# make a target based mask (use target nans)
input_mask=xr.where(np.isnan(inputs[0,:,:].values),0,1)
target_mask=xr.where(np.isnan(target[0,:,:].values),0,1)
total_mask=((input_mask+target_mask)/2).astype('int')
full_mask=xr.Dataset({
		"land_mask":(("lon","lat"),total_mask)},
	coords={"lon":target.longitude.values, "lat":target.latitude.values})
full_mask.to_netcdf('/cpc/home/gjennrich/Scripts/ML_heat/full_mask.nc',format='NETCDF4')


#!/usr/bin/python
"""
Script for Machine Learning approach to week 2 Extreme Heat
Using GEFSv12 Tmax to give a more skillful Tmax/HI forecast for hot forecasts
Target= Week 2 Tmax/HI above heat threshold (any day +8 to +14- day independent)
Future Target= Week 2 Tmax/HI above threshold (dependecy on forecast day)(each forecast day has different weights)
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
var='HI'#Tmax or HI
if var=='Tmax':
	fcst1='/cpc/gth/GTH_DATABASE/WEEK2_HEAT/GEFS_SEHOS/GEFS_v12_reforecast_airT.nc'
	obs1='/cpc/home/evan.oswald/R1_SEHOS/CDAS_1x1_reanalysis_airT.nc'
	forcast=Ds(fcst1)['max_air_temp_2m'];intal_date=Ds(fcst1)['issue_date']
	obs=Ds(obs1)['max_air_temp_2m'];obs_date=Ds(obs1)['issue_date']
elif var=='HI':
	fcst1='/cpc/gth/GTH_DATABASE/WEEK2_HEAT/GEFS_SEHOS/GEFS_v12_reforecast_HeatIndex.nc'
	obs1='/cpc/home/evan.oswald/R1_SEHOS/CDAS_1x1_reanalysis_HeatIndex.nc'
	forcast=Ds(fcst1)['max_heat_index_2m'];intal_date=Ds(fcst1)['issue_date']
	obs=Ds(obs1)['max_heat_index_2m'];obs_date=Ds(obs1)['issue_date']

hot_threshold=80

#Get datetime objects for time
forcast['time']=Get_dateime_date(intal_date)
obs['time']=Get_dateime_date(obs_date)
#Get dates for the forecast
forcast['fcst_date']=np.arange(8,15);forcast['ensemble']=np.arange(1,6)#give the day+ forecast number
forcast['longitude']=np.arange(231.0,302.0);obs=obs[:,::-1,:]#get the same lon and lats!

#test
Get_forecast_dates(forcast['time'],0)

"""
Cut down to 2000-2020, to remove 'bad' gefs data
Remove missing Obseravtions (June and July 2015)
"""
forcast=forcast.sel(time=slice(date(2000,1,1),date(2020,12,31)))

missing_idx=np.where(np.isnan(obs[2,2,:].values))[0]#known good point
missing_dates=pd.DatetimeIndex(obs.time.isel(time=missing_idx).values)
obs=obs.isel(time=np.delete(np.arange(0,obs.time.values.shape[0]),missing_idx))#remove missing dates

"""
Now take the ens mean (just to be simple) and connect forecast dates with obs dates
24255= 3465 int dates * 7 forecast dates
"""
ens_mean=False
if ens_mean:
	t_max_ens_mean=forcast.mean(dim='ensemble')
	inputs=t_max_ens_mean.stack(fcst=('time','fcst_date'))
else:
	inputs=forcast.stack(fcst=('time','fcst_date','ensemble'))
input_dates=[]
for d in tqdm(range(inputs.shape[-1])):
	date_add=pd.DatetimeIndex(inputs.time.values)[d]+timedelta(int(inputs.fcst_date.values[d]))
	#date_add=pd.DatetimeIndex(inputs.fcst.time.values)[d]
	input_dates.append(date_add)

input_dates_full=pd.DatetimeIndex(np.asarray(input_dates))
input_dates_cut=input_dates_full[~np.in1d(input_dates_full,missing_dates)]#remove missing obs days
inputs=inputs[:,:,~np.in1d(input_dates_full,missing_dates)]#remove missing obs days
inputs['fcst']=input_dates_cut;inputs=inputs.transpose('fcst','longitude','latitude')
#inputs for hot model
inputs_hot=xr.where(inputs<hot_threshold,hot_threshold-5,inputs)


target=obs.sel(time=input_dates_cut).transpose('time','longitude','latitude')#obs dates now aline with forecast dates :)
forecast_count=int(inputs.fcst.shape[0]/5)#number of indivdual forecasts
target_hot=xr.where(target<hot_threshold,hot_threshold-5,target)

#remove nan,flatten, and scale
inputs_flat=inputs.stack(point=('longitude','latitude'));inputs_hot_flat=inputs_hot.stack(point=('longitude','latitude'));target_flat=target.stack(point=('longitude','latitude'));target_hot_flat=target_hot.stack(point=('longitude','latitude'))
inputs_flat_cut=Remove_nans_1D(inputs_flat);inputs_flat_cut_hot=Remove_nans_1D(inputs_hot_flat);target_flat_cut=Remove_nans_1D(target_flat);target_hot_flat_cut=Remove_nans_1D(target_hot_flat)
scaler_input= MinMaxScaler(feature_range=(-1,1));scaler_input_hot= MinMaxScaler(feature_range=(0,1));scaler_target= MinMaxScaler(feature_range=(-1,1));scaler_target_hot= MinMaxScaler(feature_range=(0,1))
input_scaled=scaler_input.fit_transform(inputs_flat_cut);scaler_target.fit_transform(target_flat_cut)
input_scaled_hot=scaler_input_hot.fit_transform(inputs_flat_cut_hot);scaler_target_hot.fit_transform(target_hot_flat_cut)

"""
Load in the saved NNs
need to pass the custum objects arg since we have that saved as a metric
"""
def rmse_metric(y_true, y_pred):
	E = y_pred - y_true
	return K.sqrt(K.mean(E * E))
NN_all_model = keras.models.load_model(project_dir+'Scripts/ML_heat/'+var+'_ALL_Model',custom_objects={'rmse_metric':rmse_metric})
NN_hot_model = keras.models.load_model(project_dir+'Scripts/ML_heat/'+var+'_Hot_Model',custom_objects={'rmse_metric':rmse_metric})

nn_prediction=NN_all_model.predict(input_scaled)
nn_prediction_hot=NN_hot_model.predict(input_scaled_hot)


#All point model 2d map set up
nn_map_1d=scaler_target.inverse_transform(nn_prediction)
#turn back into an xarray, and unstack to get 2d map (57*25)
nn_map=inputs_flat.copy()*0
mask=get_target_mask(1).astype('bool').stack(point=('lon','lat'))
nn_map[...,mask]=nn_map_1d;nn_map=nn_map.unstack('point')

#Hot point model 2d map set up
nn_map_1d_hot=scaler_target_hot.inverse_transform(nn_prediction_hot)
#turn back into an xarray, and unstack to get 2d map (57*25)
nn_map_hot=inputs_flat.copy()*0
nn_map_hot[...,mask]=nn_map_1d_hot;nn_map_hot=nn_map_hot.unstack('point')

#input unstack for plotting
input_2d=inputs_flat.copy()*0
input_2d[...,mask]=inputs_flat_cut.values;input_2d=input_2d.unstack('point')

#target unstack for plotting- need to use the 'cut' land points in analysis
target_2d=target_flat.copy()*0
target_2d[...,mask]=target_flat_cut.values;target_2d=target_2d.unstack('point')

nn_ens_mean_=[];nn_ens_mean_hot_=[];GEFS_ens_mean_=[]
for d in tqdm(range(forecast_count)):#for each indivdual forecast day
	### find the ensemble mean
	nn_ens_mean_.append(nn_map.isel(fcst=slice(d*5,d*5+5)).mean('fcst'))
	nn_ens_mean_hot_.append(nn_map_hot.isel(fcst=slice(d*5,d*5+5)).mean('fcst'))
	GEFS_ens_mean_.append(input_2d.isel(fcst=slice(d*5,d*5+5)).mean('fcst'))
nn_ens_mean=xr.concat(nn_ens_mean_,'fcst');nn_ens_mean_hot=xr.concat(nn_ens_mean_hot_,'fcst');GEFS_ens_mean=xr.concat(GEFS_ens_mean_,'fcst')
target_obs=target_2d.isel(time=np.arange(0,target_2d.time.shape[0],5)).rename({'time': 'fcst'})#make variables the same
nn_ens_mean['fcst']=target_obs.fcst.values;nn_ens_mean_hot['fcst']=target_obs.fcst.values;GEFS_ens_mean['fcst']=target_obs.fcst.values

"""
Score All dates
"""
#Score rmse
rmse_hot=[];rmse_NN=[];rmse_GEFS=[]
for d in tqdm(range(forecast_count)):
	rmse_obs_hot=target_flat_cut.isel(time=d*5).values[np.where(target_flat_cut.isel(time=d*5).values>hot_threshold)[0]]#obs for hot points (hot NN)
	rmse_obs_NN=target_flat_cut.isel(time=d*5).values[np.where(target_flat_cut.isel(time=d*5).values>hot_threshold)[0]]#obs for hot points (hot NN)
	rmse_obs_gefs=target_flat_cut.isel(time=d*5).values[np.where(target_flat_cut.isel(time=d*5).values>hot_threshold)[0]]#obs for hot points (hot NN)
	nn_hot_forecast=nn_map_1d_hot[d*5:d*5+5,:].mean(axis=0)[np.where(target_flat_cut.isel(time=d*5).values>hot_threshold)[0]]#hot NN forecast hot points (hot NN)
	nn_forecast=nn_map_1d[d*5:d*5+5,:].mean(axis=0)[np.where(target_flat_cut.isel(time=d*5).values>hot_threshold)[0]]#NN forecast hot points (hot NN)
	GEFS_hot_forecast=inputs_flat_cut[d*5:d*5+5,:].mean(axis=0)[np.where(target_flat_cut.isel(time=d*5).values>hot_threshold)[0]]#GEFS forecast for hot points (hot NN)

	rmse_day= sqrt(mean_squared_error(nn_hot_forecast,rmse_obs_hot))
	rmse_hot.append(rmse_day)
	rmse_day= sqrt(mean_squared_error(nn_forecast,rmse_obs_NN))
	rmse_NN.append(rmse_day)
	rmse_day= sqrt(mean_squared_error(GEFS_hot_forecast,rmse_obs_gefs))
	rmse_GEFS.append(rmse_day)
print('Mean Hot NN RMSE for Hot points: ' +str(np.asarray(rmse_hot).mean()))
print('Mean All NN RMSE for Hot points: ' +str(np.asarray(rmse_NN).mean()))
print('Mean GEFS RMSE for Hot points: ' +str(np.asarray(rmse_GEFS).mean()))

#RMSE Maps
NN_Hot_maps=RMSE(xr.where(target_obs<hot_threshold,np.nan,nn_ens_mean_hot).values,xr.where(target_obs<hot_threshold,np.nan,target_obs).values,axis=0)
NN_maps=RMSE(xr.where(target_obs<hot_threshold,np.nan,nn_ens_mean).values,xr.where(target_obs<hot_threshold,np.nan,target_obs).values,axis=0)
gefs_maps=RMSE(xr.where(target_obs<hot_threshold,np.nan,GEFS_ens_mean).values,xr.where(target_obs<hot_threshold,np.nan,target_obs).values,axis=0)





"""
PLOTS:

Plot sample forecast:
-ML model output
-GEFS raw output
-Observations
"""
# input_plot=xr.where(input_2d<=hot_threshold,np.nan,input_2d)
# target_plot=xr.where(target_2d<=hot_threshold,np.nan,target_2d)
import matplotlib.pyplot as plt
import matplotlib.colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

crs = ccrs.PlateCarree(central_longitude=360)
cmap = plt.cm.hot_r#YlOrRd
def plot_background(ax1):
	ax1.set_extent([-55, -130, 21.0, 50], crs=ccrs.PlateCarree())
	ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
	ax1.yaxis.set_major_formatter(LatitudeFormatter())
	ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
	ax1.add_feature(cfeature.STATES, linewidth=0.5)
	ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
	ax1.gridlines(crs, linewidth=0.5, color='lightgray', linestyle='--')

fig, axarr = plt.subplots(4, 1, figsize=(10, 8), subplot_kw={'projection': crs}) #make the map bigger than the table
#hemispheric view
ax1=axarr[0];ax2=axarr[1];ax3=axarr[2];ax4=axarr[3]

plot_background(ax1);plot_background(ax2);plot_background(ax3);plot_background(ax4)
pick_date=-50#date(2011,7,29)
#print_skill=[hss_NN[pick_date],hss_model_ens[pick_date],hss_LR[pick_date]]
ax1.set_title('GEFSv12 '+var+' Forecast RMSE: '+str(round(rmse_GEFS[pick_date],2)))
cf1 = ax1.contourf(GEFS_ens_mean.longitude, GEFS_ens_mean.latitude, GEFS_ens_mean.isel(fcst=pick_date).transpose(), 17,cmap=cmap,levels=np.linspace(60,120,17),vmax=120, vmin=60,extend='both')
ax2.set_title('NN Corrected '+var+' Forecast RMSE: '+str(round(rmse_NN[pick_date],2)))
cf2 = ax2.contourf(nn_ens_mean.longitude,nn_ens_mean.latitude,  nn_ens_mean.isel(fcst=pick_date).transpose(),17,cmap=cmap,levels=np.linspace(60,120,17),vmax=120, vmin=60,extend='upper')
ax3.set_title('Hot NN Corrected '+var+' Forecast RMSE: '+str(round(rmse_hot[pick_date],2)))
cf3 = ax3.contourf(nn_ens_mean_hot.longitude,nn_ens_mean_hot.latitude,  nn_ens_mean_hot.isel(fcst=pick_date).transpose(),17,cmap=cmap,levels=np.linspace(60,120,17),vmax=120, vmin=60,extend='upper')
ax4.set_title('Observation: '+pd.DatetimeIndex(target_obs.fcst.values)[pick_date].strftime('%b %d %Y'))
cf4 = ax4.contourf(target_obs.longitude, target_obs.latitude,  target_obs.isel(fcst=pick_date).transpose(), 17,cmap=cmap,levels=np.linspace(60,120,17),vmax=120, vmin=60,extend='both')
cax = fig.add_axes([0.8, 0.25, 0.04, 0.5])
cbar = fig.colorbar(cf4, cax=cax,extend='both')# shrink=.5)
cbar.outline.set_edgecolor('black');cbar.outline.set_linewidth(.5)
cbar.ax.set_xlabel(var+' (F)', fontsize='x-large')
#plt.legend()
fig.suptitle('Week 2 '+var+' Forecast', fontsize='18')
#plt.savefig(var+'_NN_ens_mean_comparison.png')
plt.show()


"""
RMSE Map
"""
fig, axarr = plt.subplots(3, 1, figsize=(14, 8), subplot_kw={'projection': crs}) #make the map bigger than the table
#hemispheric view
land_mask_fix=input_2d[0,:,:].interp(latitude=np.arange(19.5,50.5,1))#fix point centering
ax1=axarr[0];ax2=axarr[1];ax3=axarr[2]
plot_background(ax1);plot_background(ax2);plot_background(ax3)
ax1.set_title('GEFSv12 '+var+' Forecast RMSE: '+str(round(np.nanmean(rmse_GEFS),2)),fontsize='18')
cf1 = ax1.pcolormesh(land_mask_fix.longitude, land_mask_fix.latitude, gefs_maps.transpose(),cmap=cmap,vmax=18, vmin=2)
ax2.set_title('NN Corrected '+var+' Forecast RMSE: '+str(round(np.nanmean(rmse_NN),2)),fontsize='18')
cf2 = ax2.pcolormesh(land_mask_fix.longitude,land_mask_fix.latitude, NN_maps.transpose(),cmap=cmap,vmax=18, vmin=2)
ax3.set_title('Hot NN Corrected '+var+' Forecast RMSE: '+str(round(np.nanmean(rmse_hot),2)),fontsize='18')
cf3 = ax3.pcolormesh(land_mask_fix.longitude,land_mask_fix.latitude, NN_Hot_maps.transpose(),cmap=cmap,vmax=18, vmin=2)
cax = fig.add_axes([0.8, 0.25, 0.04, 0.5])
cbar = fig.colorbar(cf1, cax=cax,extend='max')# shrink=.5)
cbar.outline.set_edgecolor('black');cbar.outline.set_linewidth(.5)
cbar.ax.set_ylabel('RMSE', fontsize='x-large')
fig.subplots_adjust(top=.85)
fig.suptitle('Week 2 '+var+' Forecast\n(Observed 80+)', fontsize='20')
plt.savefig(var+'_rmse_GEFS_NN_hot_Ens_Mean_observed_hot.png')
plt.show()

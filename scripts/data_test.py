import pandas as pd
idx = pd.IndexSlice
import numpy as np
import xarray as xr

from glob import glob

import os
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------
## Loading data
# A pre-processing step based on previous knowledge of the chosen forecasts dataset (WAVEWATCH III and MERCATOR) is also performed.

# General folder
data_folder_path = '/home/thiagocaminha/woww_analysis/data'
waves_folder_path = os.path.join(data_folder_path, 'waves_forecast')
currents_folder_path = os.path.join(data_folder_path, 'currents_forecast')
winds_folder_path = os.path.join(data_folder_path, 'winds_forecast')

# WAVES
# loading
os.chdir(waves_folder_path) # chdir
waves = xr.open_dataset('ww3_global_cef2_9b31_0848.nc') # load
# waves = xr.open_dataset('ww3_global_51f2_2d2c_15bb.nc') # test1

# pre-processing
waves = waves.rename({'longitude':'lon','latitude':'lat','Thgt':'swvht'})
waves['lon'] = waves['lon'] - 360
waves = waves.squeeze(drop=True)

# CURRENTS
# loading
os.chdir(currents_folder_path) # chdir
curr = xr.open_dataset('global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh_1645281307400.nc')
# curr = xr.open_dataset('global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh_1662861968722.nc') # test1

# pre-processing
curr = curr.rename({'longitude':'lon','latitude':'lat'}) # standardizing names
curr = curr.squeeze(drop=True) # squeezing dimentions to surface waters
# curr['uo'].values = curr['uo'].values*1.94384
# curr['uo'] = curr['uo'].assign_attrs(units='kt')
# curr['vo'].values = curr['vo'].values*1.94384
# curr['vo'] = curr['vo'].assign_attrs(units='kt')
curr['cvel'] = np.sqrt(curr['uo']**2 + curr['vo']**2)
curr['cvel'] = curr['cvel'].assign_attrs(units='kt')

# WIND
winds_data_path = os.path.join(winds_folder_path, 'ncep_global_a04e_6daa_7393.nc')
winds = xr.open_dataset(winds_data_path) # test1
winds = winds.rename({'longitude':'lon','latitude':'lat'})
winds['lon'] = winds['lon'] - 360
winds = winds.squeeze(drop=True)

# -------------------------------------
## Data Selection
# Let's say we want to perform the analysis at lon/lat -36.0/0., which is located in the equatorial Atlantic Ocean.

# Selection
lon_par = -36.
lat_par = 0.
waves = waves.sel(lon=lon_par,lat=lat_par,method='nearest')
curr = curr.sel(lon=lon_par,lat=lat_par,method='nearest')
winds = winds.sel(lon=lon_par,lat=lat_par,method='nearest')

# -------------------------------------
## Data Merge
# For the best functioning of the analysis function, the aimed parameters must be merged along the time dimension.

# Check time range of datasets
def get_date_range(data):
    start = data['time'].min().values.astype('str')[:19]
    end = data['time'].max().values.astype('str')[:19]
    rang = f'{start} - {end}'
    return rang
def get_time_step(data):
    start = data['time'][0].values
    end = data['time'][1].values
    time_step = pd.to_timedelta(end - start)
    return time_step

print(f'waves Time range: {get_date_range(waves)}, time step = ({get_time_step(waves)})')
print(f'curr Time range: {get_date_range(curr)}, time step = ({get_time_step(curr)})')
print(f'winds Time range: {get_date_range(winds)}, time step = ({get_time_step(winds)})')




### Time processing
# Currents data interpolation
curr_res = curr.resample(time='30T').interpolate()
time_sel = curr_res['time'][1::2]
curr_res = curr_res.sel(time=time_sel)

# Winds data interpolation
winds_res = winds.resample(time='1H').interpolate()
print(f'waves Time range: {get_date_range(waves)}, time step = ({get_time_step(waves)})')
print(f'curr Time range: {get_date_range(curr_res)}, time step = ({get_time_step(curr_res)})')
print(f'winds Time range: {get_date_range(winds_res)}, time step = ({get_time_step(winds_res)})')

### Wind parameters calculation
winds_res['wvel'] = np.sqrt(winds_res['ugrd10m']**2 + winds_res['vgrd10m']**2)

#### Merging datasets

# waves + currents data
data_wc = waves.merge(curr_res)
data_wc = data_wc[['swvht','Tper','cvel']].dropna(dim='time')
data_wcw = data_wc.merge(winds_res)[['swvht','Tper','cvel','wvel']]
data_wcw


## Exporting files
data_wc.to_netcdf('test_wave_curr.nc')
data_wcw.to_netcdf('test_wave_curr_wind.nc')

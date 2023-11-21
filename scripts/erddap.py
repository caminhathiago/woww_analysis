from erddapy import ERDDAP
import pandas as pd
import numpy as np
from typing import Union

from datetime import datetime

class WOWWErdappData():

    def __init__(self, server, protocol, dataset_id, initialize=True):

        self.server = server
        self.protocol = protocol
        self.dataset_id = dataset_id

        self.vars_rename = {'time':'date_time',
                                'ugrd10m':'uw',
                                'vgrd10m': 'vw'}

        self.vars = ['date_time','latitude','longitude',
                        'thgt','tdir','tper', # ww3 variables
                        'uw','vw'] # gfs variables
        # self.gfs_vars = ['date_time','latitude','longitude',
        #                     'uw','vw']

        if initialize:
            self.e = self.initialize_api()

    def initialize_api(self):

        e = ERDDAP(server=self.server, protocol=self.protocol)
        e.dataset_id = self.dataset_id

        e.griddap_initialize()

        return e

    def set_vars_constraints(self,
                            variables:list,
                            longitude:Union[float,list],
                            latitude:Union[float,list],
                            longitude_incr:int=10,
                            latitude_incr:int=2,
                            correct_pos:bool=True,
                            start_date:str=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            end_date:str=None):

        self.e.variables = variables
        self.e.constraints["time>="] = start_date
        if end_date:
            self.e.constraints["time<="] = end_date

        if isinstance(longitude, tuple) and isinstance(latitude, tuple):
            # longitude, latitude = self.point_position_correction(longitude, longitude_incr,
            #                                                         latitude, latitude_incr)
            self.e.constraints["longitude>="] = longitude[0]
            self.e.constraints["longitude<="] = longitude[1]
            self.e.constraints["latitude>="] = latitude[0]
            self.e.constraints["latitude<="] = latitude[1]

        elif isinstance(longitude, int) and isinstance(latitude, int):
            self.e.constraints["longitude>="] = longitude
            self.e.constraints["longitude<="] = longitude
            self.e.constraints["latitude>="] = latitude
            self.e.constraints["latitude<="] = latitude
        else:
            raise TypeError("longitude and latitude need to be both tuples (lines in space) or integers (points in space).")

        # self.e.constraints["longitude>="] = longitude[0] if isinstance(longitude,tuple) else longitude
        # self.e.constraints["longitude<="] = longitude[1] if isinstance(longitude,tuple) else longitude + 10
        # self.e.constraints["latitude>="] = latitude[0] if isinstance(latitude,tuple) else latitude
        # self.e.constraints["latitude<="] = latitude[1] if isinstance(latitude,tuple) else latitude + 2

    def grab_batch_data(self,response_type:str="pandas"):
        if response_type == 'xarray':
            return self.e.to_xarray()
        elif response_type == 'pandas':
            return self.e.to_pandas()
        else:
            raise TypeError("Response type not valid. Choose between 'xarray' or 'pandas'.")

    def point_position_correction(self, longitude, longitude_incr:int, latitude, latitude_incr:int):
        """
        For some reason, erddapy cannot handle requests for a single point in space.
        This function along with the realtime_data_slicing are a workaround to this issue.
        """

        if isinstance(longitude, tuple) and isinstance(latitude, tuple):
            return longitude, latitude

        elif isinstance(longitude, int) and isinstance(latitude, int):
            longitude_max = longitude + longitude_incr
            latitude_max = latitude + latitude_incr

            return (longitude,longitude_max), (latitude, latitude_max)
        else:
            raise TypeError("longitude and latitude need to be both tuples (area in space) or integers (point in space).")

    def realtime_data_slicing(self, data:pd.DataFrame):
        return data.isel(time=0, longitude=0, latitude=0, depth=0)

    def process_var_labels(self, data:pd.DataFrame):
         data.columns = (data.columns
                .str.lower()
                .str.replace(r'\s+\(.*\)', '', regex=True)
                )

    def rename_var_labels(self, data:pd.DataFrame):
        return data.rename(columns=self.vars_rename)

    def select_vars(self, data:pd.DataFrame):
        return data[[col for col in self.vars if col in data.columns]]

    def calc_wind_veloc(self, data:pd.DataFrame):
        return np.sqrt(data['uw']**2 + data['vw']**2)

    def calc_wind_direc(self, data:pd.DataFrame):
        wind_direction = np.degrees(np.arctan2(data['vw'], data['uw']))
        wind_direction = (wind_direction + 360) % 360
        return wind_direction

    def get_direc_quadrant(self, data:pd.DataFrame, direc_var:str):
        directional_var = data.filter(regex=direc_var)

        directions = ['N', 'NNE', 'NE', 'ENE',
                        'E', 'ESE', 'SE', 'SSE',
                        'S', 'SSW', 'SW', 'WSW',
                        'W', 'WNW', 'NW', 'NNW']

        num_directions = len(directions)
        degrees_per_direction = 360 / num_directions
        rotated_degrees = (directional_var + 11.25) % 360
        normalized_degrees = (rotated_degrees % 360 + 360) % 360

        direction_index = ((normalized_degrees // degrees_per_direction)
                            .astype(int, errors='ignore')
                            .squeeze()
                            )
        directions_map = {index: direction for index, direction in enumerate(directions)}

        return direction_index.map(directions_map)

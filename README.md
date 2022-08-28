# Workable Weather Window Analysis

A software for gathering, compiling and performing a weather window analysis with global forecast data from different sources for the purpose of aiding better decision making and operational safety in marine operations. The analysis is based on user-defined operational limits for a desired marine operation.

## To implement
- Plot operation period (operation start + time reference) [Done]
- Implement interactive dashboard (Holoviz)
  - Time-series plot
  - Table with WOWWs statistics
  - Table with times
  - Map with time range slider
- Implement relevant parameters (data selection, dashboard)
  - Wave direction
  - Winds 
- Select best workable weather window based on statistical comparisons
- Data access and processing
  - Forecast data access via API
    - [WaveWatch III](https://coastwatch.pfeg.noaa.gov/erddap/griddap/NWW3_Global_Best.html)
    - [GLOBAL_ANALYSIS_FORECAST_PHY_001_024 (Copernicus)](https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/INFORMATION)
    - [GFS Atmospheric Model](https://coastwatch.pfeg.noaa.gov/erddap/griddap/NCEP_Global_Best.html)
  - Data processing 
- Forecast validation using in situ data

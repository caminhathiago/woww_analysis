import streamlit as st
from wowwanalysis import WOWWAnalysis
import xarray as xr
import pandas as pd
from datetime import datetime, time
import os


st.set_page_config(
    page_title="Weather Window Analysis",
    page_icon="🦺",
)

# dashboard title
st.title("Weather Window Analysis")


@st.experimental_memo
def get_data() -> xr.Dataset:
    data_folder_path = '/home/thiagocaminha/woww_analysis/data'
    data_path = os.path.join(data_folder_path,'test_wave_curr.nc')
    return xr.open_dataset(data_path)

data = get_data()

# Select limit
with st.sidebar:
    ref_datetime = pd.to_datetime(data['time'].min().values.astype(str))
    op_start_date = st.date_input("Operation Start Date:", value=ref_datetime)
    op_start_time = st.time_input("Operation Start Time:", value=time(0))
    tpop = st.number_input("Estimated Operation Duration:")

    swvht_limit = st.number_input('Significant Wave Height Limit:')
    tper_limit = st.number_input('Peak Wave Period Limit:')
    cvel_limit = st.number_input('Surface Current Velocity Limit:')

st.write()
# Perform woww analysis
# op_start=pd.to_datetime('2022-02-23T00:00:00')
wf_issuance=ref_datetime

op_start = datetime.combine(op_start_date,op_start_time)

wa = WOWWAnalysis(data,
                    op_start=op_start,
                    tpop=tpop,
                    cont_factor=0.5,
                    toggle_op=True,
                    toggle_op_plot=True,
                    wf_issuance=wf_issuance,
                    swvht_limit=swvht_limit,
                    tper_limit=tper_limit,
                    cvel_limit=cvel_limit)



# creating a single-element container
# placeholder = st.empty()

# with placeholder.container():

fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    fig1 = wa.woww_analysis_interactive(width=800, height=400)

    # workable weather windows
    try:
        for col in wa.wowws:
            fig1.add_vrect(x0=wa.wowws.loc['START', col],
                            x1=wa.wowws.loc['END', col],
                                fillcolor='limegreen', opacity=0.6)
    except:
        pass

    # operation period
    fig1.add_vrect(x0=wa.op_start, x1=wa.op_end,
                    fillcolor="orange", opacity=0.5)


    st.write(fig1)

with fig_col2:
    pass
    # times_table = st.dataframe(wa.)

try:
    st.dataframe(wa.stats_table)
except:
    st.markdown(wa.status)

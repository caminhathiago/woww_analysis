"""
Workable Weather Window Analysis

Author: Thiago Caminha
Contacts:
- thiago.caminha@marinha.mil.br
- caminha.thiago@gmail.com

Project status: conceptualization

To implement:
- [DONE] one method for each dashboard plot element (e.g. stats_table,  times_table)
- wowws_allowed
    - Mark allowed/not allowed with green/red?
    - Only dispose allowed in dashboard?
- mark best woww (in plot and table)


"""

from ctypes import alignment
import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

import plotly.graph_objects as go

idx = pd.IndexSlice


class WOWWAnalysis():
    """
    Workable Weather Window (WOWW) analysis for a defined marine operation.

    """


    def __init__(self,
                data:xr.Dataset,
                tpop:pd.Timedelta,
                op_start:pd.Timedelta,
                cont_factor:float=1,
                cont_factor2:float=1,
                wf_issuance:pd.Timedelta=None,
                swvht_limit:float=None,
                tper_limit:float=None,
                cvel_limit:float=None,
                alpha_factor:float=None,
                td_woww_threshold:pd.Timedelta=pd.Timedelta(1, unit='H'),
                toggle_op:bool=False,
                toggle_op_plot:bool=False,
                figsize:tuple=(12, 3)
                ):

        self.data = data
        self.op_start = op_start
        self.tpop = pd.Timedelta(tpop, 'h')
        self.tpop_h = self.tpop_h()
        self.cont_factor = cont_factor
        self.cont_factor2 = cont_factor2
        self.wf_issuance = wf_issuance
        self.wf_from_op = self.wf_from_op()
        self.swvht_limit = swvht_limit
        self.tper_limit = tper_limit
        self.cvel_limit = cvel_limit
        self.alpha_factor = alpha_factor
        self.td_woww_threshold = td_woww_threshold
        self.figsize = figsize

        self.cont_time = self.cont_time()
        self.op_time = self.op_time()

        try:
            self.wowws_datetimes = self.wowws_datetimes()
            self.wowws = self.wowws()

            self.wowws_data = self.wowws_data()
            self.stats_table = self.stats_table()
            self.time_reference = self.time_reference()
            self.wowws_allowed = self.wowws_allowed()
            self.wowws_allowed_data = self.wowws_allowed_data()
            self.stats_table_allowed = self.stats_table_allowed()

            self.status = f"{len(self.wowws.columns)} Workable weather windows were found.\n{len(self.wowws_allowed.columns)} within the calculated Time Reference."
        except:
            self.status = "No Workable Weather Windows found for the period."



    def wowws_datetimes(self):
        """
        Returns an array of WOWWs datetimes.

        """

        wowws_datetimes = self.data['time'][(self.data['swvht'] < self.swvht_limit) & (self.data['Tper'] > self.tper_limit) & (self.data['cvel'] < self.cvel_limit)]

        return wowws_datetimes.values


    def wowws(self):
        """
        Dataframe of WOWWs with start datetimes,  end datetimes
        and duration.

        """

        if self.wowws_datetimes.size > 0:
            self._wowws_list = [self.wowws_datetimes.min()]
            for i in range(len(self.wowws_datetimes)-1):
                diff = pd.Timedelta(self.wowws_datetimes[i+1]- self.wowws_datetimes[i])
                if diff > self.td_woww_threshold:
                    self._wowws_list.append(self.wowws_datetimes[i])
                    self._wowws_list.append(self.wowws_datetimes[i+1])
                else:
                    pass
            self._wowws_list.append(self.wowws_datetimes.max())

            self._wowws_array = np.array(self._wowws_list)
            start_datetimes = self._wowws_array[0::2]
            end_datetimes = self._wowws_array[1::2]
            self._wowws_array = np.c_[start_datetimes, end_datetimes]

            wowws = pd.DataFrame(self._wowws_array.T, index=['START', 'END'])#, columns=cols_named)#, index=idexes_named)

            cols = np.arange(1, len(wowws.columns)+1).astype('str')
            append_str = 'WINDOW '
            cols_named = [append_str + window_number for window_number in cols]
            wowws.columns = cols_named
            transp_wowws = wowws.T
            transp_wowws['DURATION'] = wowws.T.diff(axis=1)['END'].values
            wowws = transp_wowws.T

            return wowws


    def wowws_allowed(self):
        """
        Dataframe of WOWWs that encompasses the calculated
        time reference for the operation (i.e. duration of woww > time reference)
        """

        # SELECT ALLOWED WOWWS BASED ON OPERATION TIME REFERENCE
        tc2 = self.cont_time * self.cont_factor2 # contingency time 2 - to make sure tr is safely within a woww

        wowws_allowed = []
        for woww_name in self.wowws.loc['DURATION'].index:
            woww_duration = self.wowws.loc['DURATION', woww_name]
            if woww_duration > (self.time_reference + tc2):
                wowws_allowed.append(woww_name)

        wowws_allowed = self.wowws[wowws_allowed]

        indexes = wowws_allowed.columns.str.replace('WINDOW ', '').astype('int')-1

        # self._wowws_allowed_array = _wowws_allowed_array[indexes]

        return wowws_allowed


    def wowws_data(self):
        """
        Dataframe of parameters data for each WOWW. Used to perform statistics.
        """

        # generate dataframe for statistics performance
        wowws_data = pd.DataFrame([], index=self.data['time'].values)

        for woww_per, i in zip(self._wowws_array, range(1, len(self._wowws_array)+1)):
            thgt = self.data['swvht'].sel(time=slice(woww_per[0], woww_per[1]))
            tper = self.data['Tper'].sel(time=slice(woww_per[0], woww_per[1]))
            cvel = self.data['cvel'].sel(time=slice(woww_per[0], woww_per[1]))
            data = np.array([thgt.values, tper.values, cvel.values]).T
            data_local = pd.DataFrame(data, index=thgt['time'].values, columns=[f'swvht_{i}', f'tper_{i}', f'cvel_{i}'])
            wowws_data = wowws_data.append(data_local)

        return wowws_data


    def wowws_allowed_data(self):
        """
        Returns data from allowed wowws
        """
        wowws_allowed_numbers = self.wowws_allowed.columns.str[-1]
        pipe_str = '|'
        reg_wowws_allowed_numbers = pipe_str.join(str(e) for e in wowws_allowed_numbers)
        wowws_allowed_data = self.wowws_data.filter(regex=reg_wowws_allowed_numbers)

        return wowws_allowed_data


    def stats_table(self):
        """
        Dataframe of descriptive statistics for each WOWW.
        """

        stats = self.wowws_data.describe().loc[['mean', '50%', 'std']]

        mean_std = stats.T['mean'].round(2).astype('str') + ' \u00B1 ' +stats.T['std'].round(3).astype('str')
        median = stats.T['50%'].round(2).astype('str')
        stats = stats.T
        stats['mean_std'] = mean_std
        stats['50%'] = median
        stats = stats[['mean_std', '50%']]
        stats = stats.T

        stats_swvht = stats.filter(regex='swvht', axis=1)
        stats_tper = stats.filter(regex='tper', axis=1)
        stats_cvel = stats.filter(regex='cvel', axis=1)

        stats_swvht = stats_swvht.rename({'mean_std':'Hs (mean \u00B1 std)', '50%': 'Hs (median)'})
        stats_swvht.columns = stats_swvht.columns.str.replace('swvht_', 'WINDOW ')

        stats_tper = stats_tper.rename({'mean_std':'Tp (mean \u00B1 std)', '50%': 'Tp (median)'})
        stats_tper.columns = stats_tper.columns.str.replace('tper_', 'WINDOW ')

        stats_cvel = stats_cvel.rename({'mean_std':'Cvel (mean \u00B1 std)', '50%': 'Cvel (median)'})
        stats_cvel.columns = stats_cvel.columns.str.replace('cvel_', 'WINDOW ')

        stats_table = self.wowws.copy()
        stats_table = pd.concat([stats_table, stats_swvht, stats_tper, stats_cvel])


        return stats_table


    def stats_table_allowed(self):
        """
        Dataframe of descriptive statistics for each WOWW that encompasses the calculated
        time reference for the operation (i.e. duration of woww > time reference)
        """

        return self.stats_table.loc[:, self.wowws_allowed.columns]


    def tpop_h(self):
        """
        Time of operational procedure in hours. Derived from Tpop.
        """

        return pd.Timedelta(self.tpop, 'h')


    def cont_time(self):
        """
        The contingency time of the marine operation. Calculated with the
        contingency factor
        """
        print(self.cont_factor)
        print(self.tpop_h)
        return pd.Timedelta(self.cont_factor*self.tpop_h, 'h')

    def wf_from_op(self):
        """
        Duration from last weather forecast issuance to the estimated
        beggining of the marine operation
        """

        wf_op_dur = self.op_start - self.wf_issuance
        return pd.Timedelta(wf_op_dur, 'h')


    def time_reference(self):
        """
        Time reference of the marine operation.
        """

        return self.tpop_h + self.cont_time # + self.wf_from_op


    def op_time(self):
        """
        Pandas Series with the operation time start datetime, end datetime
        and duration.
        """

        wf_op_dur = self.op_start - self.wf_issuance
        wf_op_dur = pd.Timedelta(wf_op_dur, 'h')


        tr = self.tpop_h + self.cont_time # + wf_op_dur
        op_end = self.op_start + tr
        self.op_end = op_end
        op_duration = op_end - self.op_start

        op_time = pd.Series({'START':self.op_start,
                            'END':op_end,
                            'DURATION':op_duration})

        return op_time


    def woww_analysis(self,
                      limits:bool=True,
                      wowws:str='all',
                      times:bool=True,
                      stats:bool=True,
                      op_period:bool=True,
                      wowws_allowed:bool=True,
                      op:bool=True):
        """
        A dashboard with relevant information regarding the WOWW analysis.
        """

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=self.figsize)
        ax.grid(b=True, which='major', color='grey', linestyle='-', axis='x', alpha=0.3)
        ax.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.3)

        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(("axes", 1.1))

        ax3.spines['left'].set_color('blue')
        ax2.spines['right'].set_color('red')
        ax3.spines['right'].set_color('green')

        ax.yaxis.label.set_color('blue')
        ax2.yaxis.label.set_color('red')
        ax3.yaxis.label.set_color('green')

        ax.tick_params(axis='y', colors='blue')
        ax2.tick_params(axis='y', colors='red')
        ax3.tick_params(axis='y', colors='green')

        swvht = self.data['swvht'].plot(ax=ax, color='blue', lw=1.5, zorder=2)
        self.data['Tper'].plot(ax=ax2, color='red', lw=1.5, zorder=2)
        self.data['cvel'].plot(ax=ax3, color='green', lw=1.5, zorder=2)

        # FORMATING AXES AND TEXTS
        # labels

        ax.set_ylabel('Hs (m)')
        ax.set_xlabel('Date-Month')
        ax2.set_title('')
        ax2.set_ylabel('Ts (s)')
        ax3.set_title('')
        ax3.set_ylabel('Current Vel (kt)')

        if limits:
            # titles
            ax.set_title(f'Workable Weather Window Analysis - Waves and Currents \nLimits: Hs = {self.swvht_limit} m,  Ts = {self.tper_limit} s,  Cvel = {self.cvel_limit}',
                    fontweight='bold', fontsize=10, pad=43)

            # horizontal lines of limits
            ax.axhline(self.swvht_limit, ls='dashed', color='blue', lw=1)
            ax2.axhline(self.tper_limit, ls='dashed', color='red', lw=1)
            ax3.axhline(self.cvel_limit, ls='dashed', color='green', lw=1)

            # legend
            custom_lines = [Line2D([0], [0], color='blue', lw=1.5),
                            Line2D([0], [0], color='blue', lw=1, ls='--'),
                            Line2D([0], [0], color='red', lw=1.5),
                            Line2D([0], [0], color='red', lw=1, ls='--'),
                            Line2D([0], [0], color='limegreen',  lw=3),
                            Line2D([0], [0], color='limegreen',  lw=1, ls='--')]
                            # Line2D([0],  [0],  color='darkgreen',  lw=3),
                            # Line2D([0],  [0],  color='darkgreen',  lw=1, ls='--')]
            legend_names = ['Hs ', f'Hs limit ({self.swvht_limit} m)', 'Ts',  f'Ts limit ({self.tper_limit} s)', 'Cvel', f'Cvel limit ({self.cvel_limit} kt)', 'OP', 'Start/End OP']
            ax2.legend(custom_lines, legend_names, loc='upper center', ncol=3, fontsize=9, framealpha=1, facecolor='white', bbox_to_anchor =(0.5,  1.25))
        else:
            ax.set_title(f'Workable Weather Window Analysis - Waves and Currents \nLimits: Hs = {self.swvht_limit} m,  Ts = {self.tper_limit} s,  Cvel = {self.cvel_limit}',
                    fontweight='bold', fontsize=10)

        # formating xticklabels and minorticks
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(ticks_loc, rotation=0, ha='center')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

        from matplotlib.ticker import AutoMinorLocator
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        if wowws == 'all':
            for col in self.wowws:
                ax.axvspan(self.wowws.loc['START', col], self.wowws.loc['END', col], facecolor='limegreen', alpha=0.6, zorder=1)
        elif wowws == 'allowed':
             for col in self.wowws_allowed:
                ax.axvspan(self.wowws_allowed.loc['START', col], self.wowws_allowed.loc['END', col], facecolor='limegreen', alpha=0.6, zorder=1)
        elif wowws == 'None':
            pass

        if op_period:
            self.plot_op_period(ax=ax)

        if times:
            self.plot_times_table(ax=ax)

        if stats:
            self.plot_stats_table(ax=ax, wowws=wowws)


    def plot_op_period(self, ax=None):
        """
        Plot op period
        """
        ax.axvspan(self.op_start, self.op_end, facecolor='darkgreen', alpha=0.5)


    def plot_stats_table(self, ax=None, wowws='all'):

        if wowws == 'all':
            stats_table_srt = self.stats_table.copy()
        elif wowws == 'allowed':
            stats_table_srt = self.stats_table_allowed.copy()

        for column in stats_table_srt.columns:
            stats_table_srt.loc[['START', 'END'], column] = pd.to_datetime(stats_table_srt.loc[['START', 'END'], column]).dt.strftime("%Y-%m-%d %H:%M")
        stats_table_srt.loc['DURATION'] = stats_table_srt.loc['DURATION'].astype('str').str.replace('days', 'days')

        stats_table_srt = stats_table_srt.reindex(['Hs (mean \u00B1 std)', 'Hs (median)', 'Tp (mean \u00B1 std)', 'Tp (median)', 'Cvel (mean \u00B1 std)', 'Cvel (median)', 'START', 'END', 'DURATION'])

        table_lenght = 0.25*stats_table_srt.shape[1]
        table_woww = ax.table(cellText=stats_table_srt.values, colLabels=stats_table_srt.columns, rowLabels=stats_table_srt.index, cellLoc='center', rowLoc='right', bbox=[0.2, -1.4, table_lenght, 1.1])

        from matplotlib.font_manager import FontProperties

        for (row, col), cell in table_woww.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))


    def plot_times_table(self, ax):
        wf_issuance = self.wf_issuance.strftime('%Y-%m-%d %H:%M')
        op_start = self.op_start.strftime('%Y-%m-%d %H:%M')
        op_end = self.op_end.strftime('%Y-%m-%d %H:%M')
        tpop = str(self.tpop).replace('days', 'days')[0:12]
        tc = str(self.cont_time).replace('days', 'days')[0:12]
        tr = str(self.time_reference).replace('days', 'days')[0:12]

        times_table_data = pd.DataFrame([tpop, tc, wf_issuance, tr, op_start, op_end])
        times_table_index = ['OP PROC TIME', 'CONT TIME', 'WF ISSUANCE', 'REF TIME', 'OP START', 'OP END']
    #     times_table_index = ['Tpop', 'Tc', 'Dprev', 'Tr', 'INÍCIO \nOPERAÇÃO', 'FIM \nOPERAÇÃO']

        from matplotlib.font_manager import FontProperties

        op_table = ax.table(cellText=times_table_data.values, rowLabels=times_table_index, cellLoc='left', rowLoc='right', bbox=[1.38, -0.005, 0.21, 1.])
        for (row, col), cell in op_table.get_celld().items():
            if (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))


    def woww_analysis_interactive(self,
                                  width,
                                  height):

        fig = go.Figure()

        # data
        trace_swvht = go.Scatter(x=self.data['time'],
                                 y=self.data['swvht'],
                                name='Hs')
        trace_tper = go.Scatter(x=self.data['time'],
                                y=self.data['Tper'],
                                name='Tp',
                                yaxis='y2',
                                line=dict(color="#FF0000"))
        trace_cvel = go.Scatter(x=self.data['time'],
                                y=self.data['cvel'],
                                name='Current Vel',
                                yaxis='y3')

        fig.add_trace(trace_swvht)
        fig.add_trace(trace_tper)
        fig.add_trace(trace_cvel)


        # Limits
        # fig.add_hline(y=self.swvht_limit, line_width=1) # bug not corrected
        # fig.add_hline(y=self.tper_limit, yref="y2", line_width=1)
        # fig.add_hline(y=self.cvel_limit, yref="y3", line_width=1)
        x0 = self.data['time'].min().values
        x1 = self.data['time'].max().values


        # fig.add_shape(type="line",
        #                 xref="paper", yref="y2",
        #                 x0=x0, y0=self.tper_limit, x1=x1, y1=self.tper_limit,
        #                 line=dict(
        #                     color="red",
        #                     dash="dash",
        #                     width=3
        #                 )
        #                 )
        swvht_limit = go.Scatter(name='Hs limit',
                            x=self.data['time'],
                            y=np.repeat(self.swvht_limit, len(self.data['time'])),
                            yaxis="y",
                            line=dict(color="#0000ff",
                                        dash="dash",
                                        width=0.5) )

        tper_limit = go.Scatter(name='Ts limit',
                            x=self.data['time'],
                            y=np.repeat(self.tper_limit, len(self.data['time'])),
                            yaxis="y2",
                            line=dict(color="#FF0000",
                                        dash="dash",
                                        width=0.5) )

        cvel_limit = go.Scatter(name='Cvel limit',
                            x=self.data['time'],
                            y=np.repeat(self.cvel_limit, len(self.data['time'])),
                            yaxis="y3",
                            line=dict(color="#006400",
                                        dash="dash",
                                        width=0.5) )

        fig.add_trace(swvht_limit)
        fig.add_trace(tper_limit)
        fig.add_trace(cvel_limit)



        fig.update_layout(width=width, height=height,
            template="simple_white",
            # split the x-axis to fraction of plots in
            # proportions
            xaxis=dict(
                domain=[0.05,1]
            ),

            # pass the y-axis title, titlefont, color
            # and tickfont as a dictionary and store
            # it an variable yaxis
            yaxis=dict(
                title="Hs (m)",
                titlefont=dict(
                    color="#0000ff"
                ),
                tickfont=dict(
                    color="#0000ff"
                )
            ),

            # pass the y-axis 2 title, titlefont, color and
            # tickfont as a dictionary and store it an
            # variable yaxis 2
            yaxis2=dict(
                title="Ts (s)",
                titlefont=dict(
                    color="#FF0000"
                ),
                tickfont=dict(
                    color="#FF0000"
                ),
                anchor="free",  # specifying x - axis has to be the fixed
                overlaying="y",  # specifyinfg y - axis has to be separated
                side="left",  # specifying the side the axis should be present
                position=0 # specifying the position of the axis
            ),

            # pass the y-axis 3 title, titlefont, color and
            # tickfont as a dictionary and store it an
            # variable yaxis 3
            yaxis3=dict(
                title="Cvel (kt)",
                titlefont=dict(
                    color="#006400"
                ),
                tickfont=dict(
                    color="#006400"
                ),
                anchor="x",     # specifying x - axis has to be the fixed
                overlaying="y",  # specifyinfg y - axis has to be separated
                side="right"  # specifying the side the axis should be present
            ),

            legend=dict(y=1.15, x=0.5, orientation="h",xanchor='center'))


        return fig

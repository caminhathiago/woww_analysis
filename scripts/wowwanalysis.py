'''
Workable Weather Window analysis functions
Author: Thiago Caminha

Project status: conceptualization
'''
from this import s
import pandas as pd
idx = pd.IndexSlice
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D



class WOWWAnalysis():

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
                td_woww_threshold:pd.Timedelta=pd.Timedelta(1,unit='H'),
                toggle_op:bool=False,
                toggle_op_plot:bool=False,
                figsize:tuple=(12,3)
                ):

        self.data = data
        self.op_start = op_start
        self.tpop = tpop
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
        self.wowws_datetimes = self.wowws_datetimes()
        self.wowws = self.wowws()
        self.cont_time = self.cont_time()
        self.wowws_data = self.wowws_data()
        self.stats_table = self.stats_table()
        self.time_reference = self.time_reference()
        self.wowws_allowed = self.wowws_allowed()




    def wowws_datetimes(self):

        wowws_datetimes = self.data['time'][(self.data['swvht'] < self.swvht_limit) & (self.data['Tper'] > self.tper_limit) & (self.data['cvel'] < self.cvel_limit)]

        return wowws_datetimes.values

    def wowws(self,data_format:str='dataframe'):

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
            self._wowws_array = np.c_[start_datetimes,end_datetimes]

            wowws = pd.DataFrame(self._wowws_array.T,index=['START','END'])#,columns=cols_named)#,index=idexes_named)

            cols = np.arange(1,len(wowws.columns)+1).astype('str')
            append_str = 'WINDOW '
            cols_named = [append_str + window_number for window_number in cols]
            wowws.columns = cols_named
            transp_wowws = wowws.T
            transp_wowws['DURATION'] = wowws.T.diff(axis=1)['END'].values
            wowws = transp_wowws.T

            return wowws

    def wowws_allowed(self):
        # SELECT ALLOWED WOWWS BASED ON OPERATION TIME REFERENCE
        tc2 = self.cont_time * self.cont_factor2 # contingency time 2 - to make sure tr is safely within a woww

        wowws_allowed = []
        for woww_name in self.wowws.loc['DURATION'].index:
            woww_duration = self.wowws.loc['DURATION',woww_name]
            if woww_duration > (self.time_reference + tc2):
                wowws_allowed.append(woww_name)

        wowws_allowed = self.wowws[wowws_allowed]

        indexes = wowws_allowed.columns.str.replace('WINDOW ','').astype('int')-1

        # self._wowws_allowed_array = _wowws_allowed_array[indexes]

        return wowws_allowed

    def wowws_data(self):

        # generate dataframe for statistics performance
        wowws_data = pd.DataFrame([],index=self.data['time'].values)
        for woww_per,i in zip(self._wowws_array,range(1,len(self._wowws_array)+1)):
            thgt = self.data['swvht'].sel(time=slice(woww_per[0],woww_per[1]))
            tper = self.data['Tper'].sel(time=slice(woww_per[0],woww_per[1]))
            cvel = self.data['cvel'].sel(time=slice(woww_per[0],woww_per[1]))
            data = np.array([thgt.values,tper.values,cvel.values]).T
            data_local = pd.DataFrame(data,index=thgt['time'].values,columns=[f'swvht {i}',f'tper {i}',f'cvel {i}'])
            wowws_data = wowws_data.append(data_local)

        return wowws_data


    def stats_table(self):

        stats = self.wowws_data.describe().loc[['mean','50%','std']]

        mean_std = stats.T['mean'].round(2).astype('str') + ' \u00B1 ' +stats.T['std'].round(3).astype('str')
        median = stats.T['50%'].round(2).astype('str')
        stats = stats.T
        stats['mean_std'] = mean_std
        stats['50%'] = median
        stats = stats[['mean_std','50%']]
        stats = stats.T

        stats_swvht = stats.filter(regex='swvht ',axis=1)
        stats_tper = stats.filter(regex='tper ',axis=1)
        stats_cvel = stats.filter(regex='cvel ',axis=1)

        stats_swvht = stats_swvht.rename({'mean_std':'Hs (mean \u00B1 std)', '50%': 'Hs (median)'})
        stats_swvht.columns = stats_swvht.columns.str.replace('swvht','WINDOW')
        stats_tper = stats_tper.rename({'mean_std':'Tp (mean \u00B1 std)', '50%': 'Tp (median)'})
        stats_tper.columns = stats_tper.columns.str.replace('tper','WINDOW')

        stats_cvel = stats_cvel.rename({'mean_std':'Cvel (mean \u00B1 std)', '50%': 'Cvel (median)'})
        stats_cvel.columns = stats_cvel.columns.str.replace('cvel','WINDOW')

        stats_table = self.wowws.copy()
        for index in stats_swvht.index:
            stats_table.loc[index] = stats_swvht.loc[index]
        for index in stats_tper.index:
            stats_table.loc[index] = stats_tper.loc[index]
        for index in stats_cvel.index:
            stats_table.loc[index] = stats_cvel.loc[index]


        return stats_table

    def tpop_h(self):
        # Returns time_pop in hours
        return pd.Timedelta(self.tpop,'h')

    def cont_time(self):
        # Calculates the contingency time
        return pd.Timedelta(self.cont_factor*self.tpop_h,'h')

    def wf_from_op(self):
        # Calculates the duration from last weather forecast issuance to the beggining of workable weather window
        wf_op_dur = self.op_start - self.wf_issuance
        return pd.Timedelta(wf_op_dur,'h') # conversion of wf_tpop_dur to hours units

    def time_reference(self):

        return self.wf_from_op + self.tpop_h + self.cont_time

    def op_time(self):

        wf_op_dur = self.op_start - self.wf_issuance # duration from last weather forecast issuance to start of workable weather window
        wf_op_dur = pd.Timedelta(wf_op_dur,'h') # conversion of wf_tpop_dur to hours units

        tr = wf_op_dur + self.tpop_h + self.cont_time # calculation of Reference Period (Tr), which corresponds to the duration of a marine operation
        op_end = self.op_start + tr

        op_duration = op_end - self.op_start

        op_time = pd.Series({'START':self.op_start,
                            'END':op_end,
                            'DURATION':op_duration})

        return op_time

    def woww_analysis(self,
                      limits:bool=True,
                      woows:bool=True,
                      stats:bool=True,
                      op:bool=True):

        fig,ax = plt.subplots(1,1,sharex=True,figsize=self.figsize)
        ax.grid(b=True, which='major', color='grey', linestyle='-',axis='x',alpha=0.3)
        ax.grid(b=True, which='minor', color='grey', linestyle='-',alpha=0.3)

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

        swvht = self.data['swvht'].plot(ax=ax,color='blue',lw=1.5,zorder=2)
        self.data['Tper'].plot(ax=ax2,color='red',lw=1.5,zorder=2)
        self.data['cvel'].plot(ax=ax3,color='green',lw=1.5,zorder=2)

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
            ax.set_title(f'Workable Weather Window Analysis - Waves and Currents \nLimits: Hs = {self.swvht_limit} m, Ts = {self.tper_limit} s, Cvel = {self.cvel_limit}',
                    fontweight='bold',fontsize=10,pad=43)

            # horizontal lines of limits
            ax.axhline(self.swvht_limit,ls='dashed',color='blue',lw=1)
            ax2.axhline(self.tper_limit,ls='dashed',color='red',lw=1)
            ax3.axhline(self.cvel_limit,ls='dashed',color='green',lw=1)

            # legend
            custom_lines = [Line2D([0], [0], color='blue', lw=1.5),
                            Line2D([0], [0], color='blue', lw=1,ls='--'),
                            Line2D([0], [0], color='red', lw=1.5),
                            Line2D([0], [0], color='red', lw=1,ls='--'),
                            Line2D([0], [0], color='limegreen', lw=3),
                            Line2D([0], [0], color='limegreen', lw=1,ls='--')]
                            # Line2D([0], [0], color='darkgreen', lw=3),
                            # Line2D([0], [0], color='darkgreen', lw=1,ls='--')]
            legend_names = ['Hs ', f'Hs limit ({self.swvht_limit} m)', 'Ts', f'Ts limit ({self.tper_limit} s)', 'Cvel', f'Cvel limit ({self.cvel_limit} kt)','OP','Start/End OP']
            ax2.legend(custom_lines, legend_names, loc='upper center',ncol=3,fontsize=9,framealpha=1,facecolor='white',bbox_to_anchor =(0.5, 1.25))
        else:
            ax.set_title(f'Workable Weather Window Analysis - Waves and Currents \nLimits: Hs = {self.swvht_limit} m, Ts = {self.tper_limit} s, Cvel = {self.cvel_limit}',
                    fontweight='bold',fontsize=10)

        # formating xticklabels and minorticks
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(ticks_loc, rotation=0, ha='center')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

        from matplotlib.ticker import AutoMinorLocator
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        if woows:
            for col in self.wowws:
                ax.axvspan(self.wowws.loc['START',col],self.wowws.loc['END',col],facecolor='limegreen',alpha=0.6,zorder=1)

        if stats:
            # plot table of woww
            stats_table_srt = self.stats_table.copy()
            for column in stats_table_srt.columns:
                stats_table_srt.loc[['START','END'],column] = pd.to_datetime(stats_table_srt.loc[['START','END'],column]).dt.strftime("%d/%m/%y %H:%M")
            stats_table_srt.loc['DURATION'] = stats_table_srt.loc['DURATION'].astype('str').str.replace('days','days')

            stats_table_srt = stats_table_srt.reindex(['Hs (mean \u00B1 std)','Hs (median)','Tp (mean \u00B1 std)','Tp (median)','Cvel (mean \u00B1 std)','Cvel (median)','START','END','DURATION'])


            table_lenght = 0.25*stats_table_srt.shape[1]
            table_woww = ax.table(cellText=stats_table_srt.values,colLabels=stats_table_srt.columns,rowLabels=stats_table_srt.index,cellLoc='center',rowLoc='right',bbox=[0.2,-1.4,table_lenght,1.1])

            from matplotlib.font_manager import FontProperties

            for (row, col), cell in table_woww.get_celld().items():
                if (row == 0) or (col == -1):
                    cell.set_text_props(fontproperties=FontProperties(weight='bold'))




        # if toggle_op:








#      # GENERATE DESCRIPTIVE STATISTICS OF Hs AND Tp
#         df_global = pd.DataFrame([],index=data_wc['time'].values)
#         for woww_per,i in zip(woww_ext_arr,range(1,len(woww_ext_arr)+1)):
#             swvht = data_wc['swvht'].sel(time=slice(woww_per[0],woww_per[1]))
#             tper = data_wc['Tper'].sel(time=slice(woww_per[0],woww_per[1]))
#             cvel = data_wc['cvel'].sel(time=slice(woww_per[0],woww_per[1]))
#             data = np.array([swvht.values,tper.values,cvel.values]).T
#             df = pd.DataFrame(data,index=swvht['time'].values,columns=[f'swvht {i}',f'Tper {i}',f'Cvel {i}'])
#             df_global = df_global.append(df)

#         stats = df_global.describe().loc[['mean','50%','std']]
#         mean_std = stats.T['mean'].round(2).astype('str') + ' \u00B1 ' +stats.T['std'].round(3).astype('str')
#         median = stats.T['50%'].round(2).astype('str')
#         stats = stats.T
#         stats['mean_std'] = mean_std
#         stats['50%'] = median
#         stats = stats[['mean_std','50%']]
#         stats = stats.T

#         stats_swvht = stats.filter(regex='swvht ',axis=1)
#         stats_tper = stats.filter(regex='Tper ',axis=1)
#         stats_cvel = stats.filter(regex='Cvel ',axis=1)

#         stats_swvht = stats_swvht.rename({'mean_std':'Hs (mean \u00B1 std)', '50%': 'Hs (median)'})
#         stats_swvht.columns = stats_swvht.columns.str.replace('swvht','WINDOW')

#         stats_tper = stats_tper.rename({'mean_std':'Tp (mean \u00B1 std)', '50%': 'Tp (median)'})
#         stats_tper.columns = stats_tper.columns.str.replace('Tper','WINDOW')

#         stats_cvel = stats_cvel.rename({'mean_std':'Cvel (mean \u00B1 std)', '50%': 'Cvel (median)'})
#         stats_cvel.columns = stats_cvel.columns.str.replace('Cvel','WINDOW')

#         woww_df_str = woww_df.copy()
#         for index in stats_swvht.index:
#             woww_df_str.loc[index] = stats_swvht.loc[index]
#         for index in stats_tper.index:
#             woww_df_str.loc[index] = stats_tper.loc[index]
#         for index in stats_cvel.index:
#             woww_df_str.loc[index] = stats_cvel.loc[index]


#         # plot table of woww
#         for column in woww_df_str.columns:
#             woww_df_str.loc[['START','END'],column] = pd.to_datetime(woww_df_str.loc[['START','END'],column]).dt.strftime("%d/%m/%y %H:%M")
#         woww_df_str.loc['DURATION'] = woww_df_str.loc['DURATION'].astype('str').str.replace('days','days')

#         woww_df_str = woww_df_str.reindex(['Hs (mean \u00B1 std)','Hs (median)','Tp (mean \u00B1 std)','Tp (median)','Cvel (mean \u00B1 std)','Cvel (median)','STAR','END','DURATION'])


#         table_lenght = 0.25*woww_df_str.shape[1]
#         table_woww = ax.table(cellText=woww_df_str.values,colLabels=woww_df_str.columns,rowLabels=woww_df_str.index,cellLoc='center',rowLoc='right',bbox=[0.2,-1.4,table_lenght,1.1])

#         from matplotlib.font_manager import FontProperties

#         for (row, col), cell in table_woww.get_celld().items():
#             if (row == 0) or (col == -1):
#                 cell.set_text_props(fontproperties=FontProperties(weight='bold'))




#         # highlight woww as axvspan
# #         for time in woww_ext:
# #             ax2.axvline(time,ls='dashed',lw=1,color='limegreen')
# #         ax2.plot(woww.values,np.repeat(data['Tper'].min().values-y_wowwbar,len(woww)),marker='s',ls='none',color='limegreen')

#         for col in woww_df:
#             ax.axvspan(woww_df.loc['START',col],woww_df.loc['END',col],facecolor='limegreen',alpha=0.6,zorder=1)

#         if toggle_op == True:
#             # OPERATION TIME REFERENCE

#             tpop = pd.Timedelta(tpop,'h') # technical operation duration in hours
#             tc = pd.Timedelta(cont_factor*tpop,'h') # contingency duration in hours

#             wf_op_dur = op_start - wf_issuance # duration from last weather forecast issuance to start of workable weather window
#             wf_op_dur = pd.Timedelta(wf_op_dur,'h') # conversion of wf_tpop_dur to hours units

#             tr = wf_op_dur + tpop + tc # calculation of Reference Period (Tr), which corresponds to the duration of a marine operation
#             op_end = op_start + tr

#             op_plot_data = [op_start,op_end]


#             # SELECT ALLOWED WOWWS BASED ON OPERATION TIME REFERENCE
#             tc2 = tc * cont2_factor # contingency time 2 - to make sure tr is safely within a woww

#             woww_allowed = []
#             for woww_name in woww_df.loc['DURATION'].index:
#                 woww_duration = woww_df.loc['DURATION',woww_name]
#                 if woww_duration > (tr + tc2):
#                     woww_allowed.append(woww_name)

#             woww_df_allowed = woww_df[woww_allowed]

#             indexes = woww_df_allowed.columns.str.replace('WINDOW ','').astype('int')-1

#             woww_ext_allowed = woww_ext_arr[indexes]

#             # SELECT BEST WOWWS BASED ON Hs STATISTICAL COMPARISONS (LOWEST Hs)
#             def select_best_woww():

#                 # TEST WHEATHER THERE'S MORE THAN ONE ALLOWED WOWW
#                 # If not, define the only available woww as the best woww
#                 if len(woww_df_allowed.columns) == 0:
#                     best_woww = []
#                     notbest_woww = []

#                 elif len(woww_df_allowed.columns) == 1:
#                     best_woww = woww_df_allowed.columns[0]
#                     notbest_woww = []

#                 else:

#                     # Create DataFrame with values of swvht of available woww's
#                     df_global_allowed = pd.DataFrame([],index=data_wc['time'].values)
#                     for woww_per,i in zip(woww_ext_allowed,range(1,len(woww_ext_allowed)+1)):
#                         swvht = data_wc['swvht'].sel(time=slice(woww_per[0],woww_per[1]))
#                         tper = data_wc['Tper'].sel(time=slice(woww_per[0],woww_per[1]))
#                         cvel = data_wc['cvel'].sel(time=slice(woww_per[0],woww_per[1]))
#                         data = np.array([swvht.values,tper.values,cvel.values]).T
#                         df = pd.DataFrame(data,index=swvht['time'].values,columns=[f'swvht {i}',f'Tper {i}',f'Cvel {i}'])
#                         df_global_allowed = df_global_allowed.append(df)

#                     # Perform statistical analysis to choose the safest woww (lowest median swvht)
#                     import itertools
#                     from math import factorial
#                     from scipy.stats import kruskal

#                     names = df_global_allowed.filter(regex='swvht',axis=1).columns
#                     list_swvht = []
#                     for col in df_global_allowed.filter(regex='swvht',axis=1).columns:
#                         list_swvht.append(df_global_allowed[col].dropna().values)

#                     alpha = 0.05
#                     if len(list_swvht) == 2:
#                         s, p = kruskal(list_swvht[0],list_swvht[1])
# #                         print(f'{tuple(names.values)}: p = {round(p,5)}')
#                     elif len(list_swvht) == 3:
#                         s, p = kruskal(list_swvht[0],list_swvht[1],list_swvht[2])
# #                         print(f'{tuple(names.values)}: p = {round(p,5)}')
#                     elif len(list_swvht) == 4:
#                         s, p = kruskal(list_swvht[0],list_swvht[1],list_swvht[2],list_swvht[3])
# #                         print(f'{tuple(names.values)}: p = {round(p,5)}')

#                     if p > alpha:
# #                         print('Samples are not statistically different')
#                           pass
#                     else:
#                         subset_len = 2
#                         subsets = itertools.combinations(list_swvht, subset_len)
#                         subsets_names = itertools.combinations(names, subset_len)
#                 #         comb = factorial(len(list_swvht))/(factorial(subset_len)*factorial(len(list_swvht)-subset_len))

#                         for subset,subset_name in zip(subsets,subsets_names):
#                             s, p = kruskal(subset[0],subset[1])
#                         #     print(f'{names}: p = {round(p,5)}')
#                             if p < 0.05:
# #                                 print(f'{subset_name}: p = {round(p,5)}')
#                                 pass

#                         medians = []
#                         for window,name in zip(list_swvht,names):
#                             median = np.median(window)
# #                             print(f'{name} median: {np.round(np.median(window),3)}')
#                             medians.append(name)
#                             medians.append(median)
#                         best_woww = pd.DataFrame(medians[1::2],index=medians[0::2],columns=['median']).T.idxmin(axis=1).values
#                         best_woww = best_woww[0].replace('swvht','WINDOW')

#                         notbest_woww = pd.DataFrame(medians[1::2],index=medians[0::2],columns=['median']).T
#                         notbest_woww = notbest_woww.loc[:, notbest_woww.columns != best_woww]
#                         notbest_woww = notbest_woww.columns.str.replace('swvht','WINDOW')

#                 return best_woww, notbest_woww

#             best_woww, notbest_woww = select_best_woww()

#             # HIGHLIGHT BEST WOWW IN WOWW TABLE

#             if best_woww:
#                 best_woww_highlight = int(best_woww[-1])-1
#                 table_woww[(0, best_woww_highlight)].set_facecolor("green")
#                 for row in range(1,10):
#                     table_woww[(row, best_woww_highlight)].set_facecolor("lightgreen")
#                 if len(notbest_woww) >= 1:
#                     notbest_woww_highlight = []
#                     for col in notbest_woww:
#                         notbest_woww_highlight_temp = int(col[-1])
#                         notbest_woww_highlight.append(notbest_woww_highlight_temp)
#                     for notbest_woww_el in notbest_woww_highlight:
#                         table_woww[(0, notbest_woww_el)].set_facecolor("lightgreen")
#                 elif not len(notbest_woww) >= 1:
#                     for (row, col), cell in table_woww.get_celld().items():
#                         if (row == 0):
#                             cell.set(facecolor='lightcoral')
#                     table_woww[(0, best_woww_highlight)].set_facecolor("green")
#             if not best_woww:
#                 for (row, col), cell in table_woww.get_celld().items():
#                         if (row == 0):
#                             cell.set(facecolor='lightcoral')

# #             elif notbest_woww:
# #                 best_woww_highlight = int(best_woww[-1])-1
# #                 notbest_woww_highlight = []
# #                 for col in notbest_woww_highlight:
# #                     notbest_woww_highlight_temp = int(col[-1])-1
# #                     notbest_woww_highlight.append(notbest_woww_highlight_temp)

# #                 table_woww[(0, best_woww_highlight)].set_facecolor("green")
# #                 for notbest_woww in notbest_woww_highlight:
# #                     table_woww[(0, notbest_woww)].set_facecolor("lightgreen")




#             if toggle_op_plot == True:
#                 # plot OP timeline
# #                 op_plot = ax2.plot(op_plot_data,np.repeat(data['Tper'].min().values-y_wowwbar,len(op_plot_data)),color='darkgreen',lw=7)
# #                 ax2.axvline(op_start,ls='dashed',lw=1,color='darkgreen')
# #                 ax2.axvline(op_end,ls='dashed',lw=1,color='darkgreen')
#                 op_plot_axvspan = ax.axvspan(op_plot_data[0],op_plot_data[1],facecolor='darkgreen',alpha=0.5)


#                 # plot OP table
#                 wf_issuance = wf_issuance.strftime('%d/%m/%Y %H:%M')
#                 op_start = op_start.strftime('%d/%m/%Y %H:%M')
#                 op_end = op_end.strftime('%d/%m/%Y %H:%M')
#                 tpop = str(tpop).replace('days','days')[0:12]
#                 tc = str(tc).replace('days','days')[0:12]
#                 tr = str(tr).replace('days','days')[0:12]

#                 op_table_data = pd.DataFrame([tpop,tc,wf_issuance,tr,op_start,op_end])
#                 op_table_index = ['OP PROC TIME','CONT TIME','WF ISSUANCE','REF TIME','OP START','OP END']
#             #     op_table_index = ['Tpop','Tc','Dprev','Tr','INÍCIO \nOPERAÇÃO','FIM \nOPERAÇÃO']

#                 op_table = ax.table(cellText=op_table_data.values,rowLabels=op_table_index,cellLoc='left',rowLoc='right',bbox=[1.45,-0.005,0.21,1.])
#                 for (row, col), cell in op_table.get_celld().items():
#                     if (col == -1):
#                         cell.set_text_props(fontproperties=FontProperties(weight='bold'))

#     else:
#         # plot warning that no woww are available
#         props = dict(boxstyle='square', facecolor='white',edgecolor='red', alpha=1)
#         ax.text(0.23,-0.47,'NO WORKABLE WINDOWS ALLOWED',fontsize=15, fontweight='bold', transform=ax.transAxes,bbox=props)




#     # labels
#     ax.set_title(f'Workable Weather Window Analysis - Waves and Currents \nLimits: Hs = {swvht_limit} m, Ts = {tper_limit} s, Cvel = {cvel_limit}',
#                  fontweight='bold',fontsize=10,pad=43)
#     ax.set_ylabel('Hs (m)')
#     ax.set_xlabel('Date-Month')
#     ax2.set_title('')
#     ax2.set_ylabel('T (s)');
#     ax3.set_title('')
#     ax3.set_ylabel('Current Vel (kt)');

#     # legend
#     custom_lines = [Line2D([0], [0], color='blue', lw=1.5),
#                    Line2D([0], [0], color='blue', lw=1,ls='--'),
#                    Line2D([0], [0], color='red', lw=1.5),
#                    Line2D([0], [0], color='red', lw=1,ls='--'),
#                    Line2D([0], [0], color='limegreen', lw=3),
#                    Line2D([0], [0], color='limegreen', lw=1,ls='--'),
#                    Line2D([0], [0], color='darkgreen', lw=3),
#                    Line2D([0], [0], color='darkgreen', lw=1,ls='--')]
#     legend_names = ['Hs ', f'Hs limit ({swvht_limit} m)', 'Ts', f'Ts limit ({tper_limit} s)', 'Cvel', f'Cvel limit ({cvel_limit} kt)','JT','Start/End WW','OP','Start/End OP']
#     ax2.legend(custom_lines, legend_names, loc='upper center',ncol=4,fontsize=9,framealpha=1,facecolor='white',bbox_to_anchor =(0.5, 1.25))

#     #
#     ymin1,ymax1 = ax.get_ylim()
#     ax.set_ylim(ymin1*0.9,ymax1*1.2)
#     ymin2,ymax2 = ax2.get_ylim()
# #     ax2.set_ylim(ymin2*0.9,ymax2*1.2)

#     # formating xticklabels and minorticks
#     ticks_loc = ax.get_xticks().tolist()
#     ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
#     ax.set_xticklabels(ticks_loc, rotation=0, ha='center')
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

#     from matplotlib.ticker import AutoMinorLocator
#     ax.xaxis.set_minor_locator(AutoMinorLocator(2))

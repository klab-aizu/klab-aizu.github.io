import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8

width = 0.15  # width of a bar
n_bench = 13
case_len = 4
column_names = ['bench', 'T_ref', 'defected',
                'virtual', 'serial', 'disable', 'redundant']
# bench_names = ['blackscholes', 'canneal', 'dedup', 'facesim', 'ferret', 'fluidanime', 'swaptions', 'x264', 'hotspot-5%', 'hotspot-10%', 'matrix', 'transpose', 'uniform']
bench_names = ['blackscholes', 'canneal', 'dedup', 'facesim', 'ferret', 'fluidanime',
               'swaptions', 'x264', 'hotspot-5%', 'hotspot-10%', 'matrix', 'transpose', 'uniform']

WB_none = pd.DataFrame(columns=column_names)
WB_int = pd.DataFrame(columns=column_names)
WB_ext = pd.DataFrame(columns=column_names)
WB_hyb = pd.DataFrame(columns=column_names)
WB_hot = pd.DataFrame(columns=column_names)
FF_none = pd.DataFrame(columns=column_names)
FF_int = pd.DataFrame(columns=column_names)
FF_ext = pd.DataFrame(columns=column_names)
FF_hyb = pd.DataFrame(columns=column_names)
FF_hot = pd.DataFrame(columns=column_names)

my_colors = ['brown', 'red', 'green', 'blue', 'purple', 'gray']


def read_dat():
    for test_case in range(n_bench):
        start_indx = test_case*case_len+1
        # remove last one since it is last layer
        end_indx = (test_case+1)*case_len
        weight_none = pd.read_csv('Tref/HC_weight-based_w_red_none.csv', delimiter=',',
                                  skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        weight_int = pd.read_csv('Tref/HC_weight-based_w_red_int.csv', delimiter=',',
                                 skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        weight_ext = pd.read_csv('Tref/HC_weight-based_w_red_ext.csv', delimiter=',',
                                 skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        weight_hyb = pd.read_csv('Tref/HC_weight-based_w_red_hyb.csv', delimiter=',',
                                 skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        weight_hot = pd.read_csv('Tref/HC_weight-based_w_red_hot-cluster.csv', delimiter=',',
                                 skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        ff_none = pd.read_csv('Tref/HC_ford-fulkerson_w_red_none.csv', delimiter=',',
                              skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        ff_int = pd.read_csv('Tref/HC_ford-fulkerson_w_red_int.csv', delimiter=',',
                             skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        ff_ext = pd.read_csv('Tref/HC_ford-fulkerson_w_red_ext.csv', delimiter=',',
                             skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        ff_hyb = pd.read_csv('Tref/HC_ford-fulkerson_w_red_hyb.csv', delimiter=',',
                             skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        ff_hot = pd.read_csv('Tref/HC_ford-fulkerson_w_red_hot-cluster.csv', delimiter=',',
                             skiprows=lambda x: x not in range(start_indx, end_indx) and x != 0, usecols=column_names)
        print(weight_hyb)
        # average?
        WB_none.loc[weight_none['bench'][0]] = weight_none.mean()
        WB_int.loc[weight_int['bench'][0]] = weight_int.mean()
        WB_ext.loc[weight_ext['bench'][0]] = weight_ext.mean()
        WB_hyb.loc[weight_hyb['bench'][0]] = weight_hyb.mean()
        WB_hot.loc[weight_hot['bench'][0]] = weight_hot.mean()

        FF_none.loc[weight_none['bench'][0]] = weight_none.mean()
        FF_int.loc[weight_int['bench'][0]] = weight_int.mean()
        FF_ext.loc[weight_ext['bench'][0]] = weight_ext.mean()
        FF_hyb.loc[weight_hyb['bench'][0]] = weight_hyb.mean()
        FF_hot.loc[weight_hot['bench'][0]] = weight_hot.mean()
    print(WB_int)


# plt.show()
read_dat()
v_alpha = .7
ax1 = plt.subplot(511)
ax1.bar(np.arange((n_bench))+width*-2, WB_none['defected'], width=width,
        align='center', alpha=v_alpha, label='injected', color=my_colors[0])
ax1.bar(np.arange((n_bench))+width*-2, WB_none['redundant'], width=width,
        align='center', alpha=v_alpha, label='none', color=my_colors[1])
ax1.bar(np.arange((n_bench))+width*-1, WB_int['redundant'], width=width,
        align='center', alpha=v_alpha, label='int.red.', color=my_colors[2])
ax1.bar(np.arange((n_bench))+width*0, WB_ext['redundant'], width=width,
        align='center', alpha=v_alpha, label='ext.red.', color=my_colors[3])
ax1.bar(np.arange((n_bench))+width*1, WB_hyb['redundant'], width=width,
        align='center', alpha=v_alpha, label='hyb.red.', color=my_colors[4])
ax1.bar(np.arange((n_bench))+width*2, WB_hot['redundant'], width=width,
        align='center', alpha=v_alpha, label='irr. red. (HotCluster)', color=my_colors[5])

ax0 = ax1.twinx()
ax0.set_ylabel('$T_{ref}$')
ax0.plot(np.arange((n_bench)), WB_none['T_ref'], '*--', color='red')

ax1.set_ylabel("Redundancy Ratio", fontsize=8)
ax1.legend(
    bbox_to_anchor=(0.0, 1.0, 1.0, 0.102),
    loc=3,
    ncol=7,
    mode="expand",
    borderaxespad=0.0,
)

# plt.xticks(np.arange((n_bench)), bench_names)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(512, sharex=ax1)
ax2.bar(np.arange((n_bench))+-2*width,
        WB_none['disable']/100, width=width,  alpha=v_alpha, label='none', color=my_colors[1])
ax2.bar(np.arange((n_bench))+-1*width,
        WB_int['disable']/100, width=width,  alpha=v_alpha, label='int. red.', color=my_colors[2])
ax2.bar(np.arange((n_bench))+0*width,
        WB_ext['disable']/100, width=width,  alpha=v_alpha, label='ext. red.', color=my_colors[3])
ax2.bar(np.arange((n_bench))+1*width,
        WB_hyb['disable']/100, width=width,  alpha=v_alpha, label='hyb. red.', color=my_colors[4])
ax2.bar(np.arange((n_bench))+2*width, WB_hot['disable']/100, width=width,
        alpha=v_alpha, label='irr. red. (HotCluster)', color=my_colors[5])
# plt.xticks(np.arange((n_bench)), bench_names)
ax2.set_ylabel("Disable Router", fontsize=8)

plt.setp(ax2.get_xticklabels(), visible=False)
# ax2.legend(
#     bbox_to_anchor=(0.0, 1.02, 1.02, 0.102),
#     loc=3,
#     ncol=7,
#     mode="expand",
#     borderaxespad=0.0,
# )
ax3 = plt.subplot(513, sharex=ax1)

ax3.bar(np.arange((n_bench))+-2*width,
        WB_none['serial']/100, width=width,  alpha=v_alpha, label='none', color=my_colors[1])
ax3.bar(np.arange((n_bench))+-1*width,
        WB_int['serial']/100, width=width,  alpha=v_alpha, label='int. red.', color=my_colors[2])
ax3.bar(np.arange((n_bench))+0*width,
        WB_ext['serial']/100, width=width, alpha=v_alpha, label='ext. red.', color=my_colors[3])
ax3.bar(np.arange((n_bench))+1*width,
        WB_hyb['serial']/100, width=width,  alpha=v_alpha, label='hyb. red.', color=my_colors[4])
ax3.bar(np.arange((n_bench))+2*width, WB_hot['serial']/100, width=width,
        alpha=v_alpha, label='irr. red. (HotCluster)', color=my_colors[5])
# plt.xticks(np.arange((n_bench)), bench_names)
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set_ylabel("Serialziation Router", fontsize=8)

ax4 = plt.subplot(514, sharex=ax1)

ax4.bar(np.arange((n_bench))+-2*width,
        WB_none['virtual']/100, width=width,  alpha=v_alpha, label='none', color=my_colors[1])
ax4.bar(np.arange((n_bench))+-1*width,
        WB_int['virtual']/100, width=width,  alpha=v_alpha, label='int. red.', color=my_colors[2])
ax4.bar(np.arange((n_bench))+0*width,
        WB_ext['virtual']/100, width=width,  alpha=v_alpha, label='ext. red.', color=my_colors[3])
ax4.bar(np.arange((n_bench))+1*width,
        WB_hyb['virtual']/100, width=width,  alpha=v_alpha, label='hyb. red.', color=my_colors[4])
ax4.bar(np.arange((n_bench))+2*width, WB_hot['virtual']/100, width=width,
        alpha=v_alpha, label='irr. red. (HotCluster)', color=my_colors[5])
# plt.xticks(np.arange((n_bench)), bench_names)
plt.setp(ax4.get_xticklabels(), visible=False)
ax4.set_ylabel("Virtualized Router", fontsize=8)

ax5 = plt.subplot(515, sharex=ax1)

ax5.bar(np.arange((n_bench))+-2*width, (100-WB_none['disable']-WB_none['serial'] -
                                        WB_none['virtual'])/100, width=width,  alpha=v_alpha, label='none', color=my_colors[1])
ax5.bar(np.arange((n_bench))+-1*width, (100-WB_int['disable'] - WB_int['serial'] -
                                        WB_int['virtual'])/100, width=width,  alpha=v_alpha, label='int. red.', color=my_colors[2])
ax5.bar(np.arange((n_bench))+0*width, (100-WB_ext['disable'] - WB_ext['serial'] -
                                       WB_ext['virtual'])/100, width=width,  alpha=v_alpha, label='ext. red.', color=my_colors[3])
ax5.bar(np.arange((n_bench))+1*width, (100-WB_hyb['disable'] - WB_hyb['serial'] -
                                       WB_hyb['virtual'])/100, width=width,  alpha=v_alpha, label='hyb. red.', color=my_colors[4])
ax5.bar(np.arange((n_bench))+2*width, (100-WB_hot['disable'] - WB_hot['serial'] - WB_hot['virtual']
                                       )/100, width=width,  alpha=v_alpha, label='irr. red. (HotCluster)', color=my_colors[5])
plt.xticks(np.arange((n_bench)), bench_names)
plt.xticks(rotation=30)
ax5.set_ylabel("Normal Router", fontsize=8)

# plt.show()
plt.savefig("fig_eva_T_ref_2.pdf", bbox_inches="tight", format="pdf", dpi=1000)

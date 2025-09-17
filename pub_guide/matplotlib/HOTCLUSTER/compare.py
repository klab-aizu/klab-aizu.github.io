import pandas as pd
import numpy as np
import altair as alt
XOrder = { "CPWI-no-red.":0, "CPWI-red.":1, "SAWI-int.-red.":2, "SAWI-ext.-red.": 3, "SAWI-hyb.-red.":4, "FF-int.-red.":5, "FF-ext.-red.":6, "FF-hyb.-red.":7, "99":99, "100":100}
StackOrder = { 'normal':0, 'virtual':1, 'serial':2, 'disable': 3}


def prep_df(df, name):
    df = df.stack().reset_index()
    df.columns = ['c1', 'DF', 'values']
    df['c2'] = name
    df['StackOrder'] = df['DF'].map(StackOrder)#pd.Series(StackOrder[df['DF']], index = df.index)
    df['XOrder'] = df['c2'].map(XOrder) #pd.Series(XOrder[df['c2']], index = df.index)
    return df
b = [alt.Chart(), alt.Chart(), alt.Chart()]
for test_case in range(3):


    start_indx = test_case*11+1
    end_indx = test_case*11+12

    d1 = pd.read_csv('trace_wo_red.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0, usecols=['injected','virtual','serial','disable'])
    d2 = pd.read_csv('trace_w_red.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])
    d3 = pd.read_csv('weight-based_w_red_int.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])
    d4 = pd.read_csv('weight-based_w_red_ext.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])
    d5 = pd.read_csv('weight-based_w_red_hyb.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])
    d6 = pd.read_csv('ford-fulkerson_w_red_int.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])
    d7 = pd.read_csv('ford-fulkerson_w_red_ext.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])
    d8 = pd.read_csv('ford-fulkerson_w_red_hyb.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])
    
    # d99 = pd.read_csv('99.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])
    # d100 = pd.read_csv('100.csv',delimiter=',',index_col=['injected'], skiprows = lambda x: x not in range(start_indx, end_indx) and x !=0,usecols=['injected','virtual','serial','disable'])

    # print(d1)

    d1['normal'] = pd.Series(100-d1['virtual']-d1['serial']-d1['disable'], index=d1.index)
    d1 = (prep_df(d1[['normal','virtual','serial','disable']], "CPWI-no-red."))

    d2['normal'] = pd.Series(100-d2['virtual']-d2['serial']-d2['disable'], index=d2.index)
    d2 = (prep_df(d2[['normal','virtual','serial','disable']], "CPWI-red."))

    d3['normal'] = pd.Series(100-d3['virtual']-d3['serial']-d3['disable'], index=d3.index)
    d3 = (prep_df(d3[['normal','virtual','serial','disable']], "SAWI-int.-red."))

    d4['normal'] = pd.Series(100-d4['virtual']-d4['serial']-d4['disable'], index=d4.index)
    d4 = (prep_df(d4[['normal','virtual','serial','disable']], "SAWI-ext.-red."))

    d5['normal'] = pd.Series(100-d5['virtual']-d5['serial']-d5['disable'], index=d5.index)
    d5 = (prep_df(d5[['normal','virtual','serial','disable']], "SAWI-hyb.-red."))

    d6['normal'] = pd.Series(100-d6['virtual']-d6['serial']-d6['disable'], index=d6.index)
    d6 = (prep_df(d6[['normal','virtual','serial','disable']], "FF-int.-red."))

    d7['normal'] = pd.Series(100-d7['virtual']-d7['serial']-d7['disable'], index=d7.index)
    d7 = (prep_df(d7[['normal','virtual','serial','disable']], "FF-ext.-red."))

    d8['normal'] = pd.Series(100-d8['virtual']-d8['serial']-d8['disable'], index=d8.index)
    d8 = (prep_df(d8[['normal','virtual','serial','disable']], "FF-hyb.-red."))

    # d99['normal'] = pd.Series(100-d99['virtual']-d99['serial']-d99['disable'], index=d99.index)
    # d99 = (prep_df(d99[['normal','virtual','serial','disable']], "99"))
    # d100['normal'] = pd.Series(100-d100['virtual']-d100['serial']-d100['disable'], index=d100.index)
    # d100 = (prep_df(d100[['normal','virtual','serial','disable']], "100"))

    dx = pd.concat([d1, d2, d3,d4,d5,d6,d7,d8])
    # print("df")

    # print(dx)


    b[test_case] = alt.Chart(dx, height=80, width=80).mark_bar().encode(

        # tell Altair which field to group columns on
        x=alt.X('c2:N',  title=None, sort=alt.EncodingSortField(field="XOrder", order='ascending') ),

        # tell Altair which field to use as Y values and how to calculate
        y=alt.Y('sum(values):Q',
            axis=alt.Axis(
                grid=False,
                title="Percentage (%)")),

        # tell Altair which field to use to use as the set of columns to be  represented in each group
        column=alt.Column('c1:N',  title='Defect Rate (%)' if (test_case==0) else None),
        order=alt.Order("StackOrder:N", sort="ascending"),
        # tell Altair which field to use for color segmentation 
        color=alt.Color('DF:N', sort = ['normal', 'virtual', 'serial', 'disable'],
                scale=alt.Scale(
                    # make it look pretty with an enjoyable color pallet
                    range=['#f6ce04', '#ff0c5c','#afaff9',"#000000"],
                ), 
            ))
    
alt.vconcat(b[0], b[1], b[2]).resolve_scale(color='independent').configure_view(
    stroke=None
).save('barchart_compare.html', embed_options={'renderer':'svg'})
# b.save('barchart_'+str(test_case)+'.html', embed_options={'renderer':'svg'})
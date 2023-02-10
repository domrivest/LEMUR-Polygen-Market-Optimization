""" Results of market analysis and matplotlib plots
Author : Dominic Rivest
Analysis of the 27 market scenarios
Created in 2022 """


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy_financial import irr

# Market scenarios definition
seriesOpti =[]
for ElecScen in ['eL', 'eM', 'eH']: # Low, median and high electricity market price
    for MeOHScen in ['mL', 'mM', 'mH']: # Low, median and high methanol market price
        for H2Scen in ['h2L', 'h2M', 'h2H']: # Low, median and high hydrogen market price
            seriesOpti.append(ElecScen+MeOHScen+H2Scen)

# Initial values and parameters

years = range(2022, 2037) # 15 years Equipment life (2037 not included)
discountRate = 0.1 # 10 % base discount rate
OM = 0.055 # 5.5% Of FIC for operation and maintenance
showPLT = 1 # 1 show matplotlib graphs
tau = 0.25 # 15 minute timesteps
TRNSYSInfo = pd.read_excel('Serre_plant-condensation-control_building.xlsx')  # Greenhouse heat demand
heatDemandIni = TRNSYSInfo['MW pour 20 000 m2']*1000*tau # Conversion from MW to kW (mettre *tau pour kWh)
nominalPower = np.multiply(range(1600, 8001, 1600), tau) # System nominal power (kW*15minutes, kWh)
inflationArray = np.ones(15)*1.0304 # Canadian mean inflation rate 2017-2021
inflationArray[0] = 1.0684 # Adjusted for 2022 Jan-Sep mean inflation value for first year

for i in range(len(inflationArray)): # Cumulative inflation calculation
    if i > 0:
        inflationArray[i] = inflationArray[i-1]*inflationArray[i]

sumHeatDemand = sum(heatDemandIni)# Demande totale en chaleur pour l'année [kW]
meanHeatDemand = np.mean(heatDemandIni)/tau # Moyenne de la demande en chaleur pour l'année [kW]
maxHeatDemand = np.max(heatDemandIni)/tau # Max ""

# System conversion efficiencies depending on the operating mode (1-max elec, 2 - max MeOH, 3 - max H2, 4 - Max Thermal)

etaSys = {'Elec': {1: 0.161, 2: 0, 3: 0, 4: 0},
            'MeOH': {1: 0.122, 2:0.141, 3: 0.111, 4: 0.122},
            'H2': {1: 0, 2: 0, 3: 0.319, 4:0},
            'Heat': {1: 0.566, 2: 0.708, 3: 0.420, 4: 0.717}}

# df results loading

# create empty list
df_optimalValues = []
df_optimalValuesTotal = []
 
# append datasets into the list, sum operating modes, calculate heat excess
for i in range(len(seriesOpti)):
    temp_df = pd.read_csv("CSV/optimalValues"+seriesOpti[i]+"(8000kW).csv")
    temp_df['Heat demand (kWh)'] = heatDemandIni[0:len(heatDemandIni)-1]
    temp_df['Biomass heat (kWh)'] = temp_df['Biomass']*temp_df['operatingMode1']*etaSys['Heat'][1]+temp_df['Biomass']*temp_df['operatingMode2']*etaSys['Heat'][2]\
        +temp_df['Biomass']*temp_df['operatingMode3']*etaSys['Heat'][3]+temp_df['Biomass']*temp_df['operatingMode4']*etaSys['Heat'][4]
    df_optimalValues.append(temp_df)
    temp_dfT = pd.read_csv("CSV/optimalValuesTotal"+seriesOpti[i]+"(8000kW).csv")
    temp_dfT['Max electricity'] = sum(temp_df['operatingMode1'])/(len(heatDemandIni)-1)
    temp_dfT['Max MeOH'] = sum(temp_df['operatingMode2'])/(len(heatDemandIni)-1)
    temp_dfT['Max H2'] = sum(temp_df['operatingMode3'])/(len(heatDemandIni)-1)
    temp_dfT['Max heat'] = sum(temp_df['operatingMode4'])/(len(heatDemandIni)-1)
    temp_dfT['Biomass heat (kWh)'] = sum(temp_df['Biomass heat (kWh)'])
    temp_dfT['Excess heat (kWh)'] = sum(temp_df['Biomass heat (kWh)'][heatDemandIni[0:len(heatDemandIni)-1]<temp_df['Biomass heat (kWh)']]
    -heatDemandIni[0:len(heatDemandIni)-1][heatDemandIni[0:len(heatDemandIni)-1]<temp_df['Biomass heat (kWh)']])
    temp_dfT['Heat demand (kWh)'] = sum(temp_df['Heat demand (kWh)'])
    df_optimalValuesTotal.append(temp_dfT)

df_optimalValuesTotal = pd.concat(df_optimalValuesTotal[:], ignore_index=True) # Merge of the df in one


# Ajout et correction des valeurs de somme

df_optimalValuesTotal['Particularity'] = seriesOpti
df_optimalValuesTotal['BiomassCost'] = -df_optimalValuesTotal['BiomassCost']
df_optimalValuesTotal['MeOHCostIn'] = -df_optimalValuesTotal['MeOHCostIn']

# Lecture des fichiers de EC
df_polygenEC = pd.read_csv('CSV/polygenEC.csv')

# Annual cash flow and cumulative NPV calculations

def cashFlow(df_EC, df_optimalTotal, years, discountRate, OM, inflationArray):
    temp_df = pd.DataFrame()
    temp_df.index = years
    temp_df['Year'] = years
    for year in years:
        for j in range(0, len(df_optimalTotal['Particularity'])):
            temp_df.loc[year, 'Cash Flow' + str(df_optimalTotal.loc[j, 'Particularity'])] = (df_optimalTotal.loc[j, 'ObjectiveFunction']+OM*df_EC.loc[4, 'Total FIC'])*inflationArray[year-years[0]]/(1 + discountRate)**(year-years[0]+1)
            temp_df.loc[year, 'Cumulative NPV' + str(df_optimalTotal.loc[j, 'Particularity'])] = df_EC.loc[4, 'Total FIC'] \
                +sum(temp_df.loc[years[0]:year, 'Cash Flow' + str(df_optimalTotal.loc[j, 'Particularity'])])
    return temp_df


df_polygenCashFlow = cashFlow(df_polygenEC, df_optimalValuesTotal, years, discountRate, OM, inflationArray)

dfNPV = pd.DataFrame()
# Adding information on product prices type (low, median, high)
for i in range(len(seriesOpti)):
    dfNPV.loc[i, 'SeriesOpti'] = seriesOpti[i]
    dfNPV.loc[i, 'Cumulative NPV'] = df_polygenCashFlow.loc[2036, str('Cumulative NPV'+seriesOpti[i])]
    if seriesOpti[i].find('eL', 0, 2) == 0:
        dfNPV.loc[i, 'Electricity Price'] = 'Low'
    elif seriesOpti[i].find('eM', 0, 2) == 0:
        dfNPV.loc[i, 'Electricity Price'] = 'Median'
    elif seriesOpti[i].find('eH', 0, 2) == 0:
        dfNPV.loc[i, 'Electricity Price'] = 'High'
    
    if seriesOpti[i].find('mL') == 2:
        dfNPV.loc[i, 'Methanol Price'] = 'Low'
    elif seriesOpti[i].find('mM') == 2:
        dfNPV.loc[i, 'Methanol Price'] = 'Median'
    elif seriesOpti[i].find('mH') == 2:
        dfNPV.loc[i, 'Methanol Price'] = 'High'
    
    if seriesOpti[i].find('h2L') == 4:
        dfNPV.loc[i, 'Hydrogen Price'] = 'Low'
        dfNPV.loc[i, 'Color'] = 'cyan'
    elif seriesOpti[i].find('h2M') == 4:
        dfNPV.loc[i, 'Hydrogen Price'] = 'Median'
        dfNPV.loc[i, 'Color'] = 'turquoise'
    elif seriesOpti[i].find('h2H') == 4:
        dfNPV.loc[i, 'Hydrogen Price'] = 'High'
        dfNPV.loc[i, 'Color'] = 'teal'



# IRR calculation
for i in range(len(df_optimalValuesTotal)):
    dfNPV.loc[i, 'Particularity'] = df_optimalValuesTotal.loc[i, 'Particularity']
    dfNPV.loc[i, 'IRR'] = irr(np.concatenate(([-df_polygenEC.loc[4, 'Total FIC']] , np.multiply(np.ones(len(years)), -df_optimalValuesTotal.loc[i, 'ObjectiveFunction'])-OM*df_polygenEC.loc[4, 'Total FIC'])))

# Inversion du signe de la NPV chiffres
dfNPV['Cumulative NPV'] = np.multiply(dfNPV['Cumulative NPV'], -1)
dfNPV = dfNPV.sort_values('Cumulative NPV')

# Graphique NPV vs market scenario[df_polygenEC.loc[4, 'Total FIC'] , np.multiply(np.ones(len(years)), df_optimalValuesTotal.loc[i, 'ObjectiveFunction'])]

fig, ax = plt.subplots()

ax.bar(x='SeriesOpti', height='Cumulative NPV', color='Color', data=dfNPV, alpha=0.5)
# Rotating X-axis labels
for tick in ax.get_xticklabels():
    tick.set_rotation(75)
#ax.legend(['Electricity', 'MeOH', 'Hydrogen', 'Biomass', 'MeOH (peaking burner)', 'Fixed OPEX', 'Annual cash flow sum'])
    # Ajout des valeurs en bout de barre
    for i in range(len(dfNPV['Cumulative NPV'])):
        if dfNPV['Cumulative NPV'][i] <= 0:
            ax.annotate(str(round(np.divide(dfNPV['Cumulative NPV'][i], 10**6), 2)), xy=(dfNPV['SeriesOpti'][i],dfNPV['Cumulative NPV'][i]), ha='center', va='top')
        else:
            ax.annotate(str(round(np.divide(dfNPV['Cumulative NPV'][i], 10**6), 2)), xy=(dfNPV['SeriesOpti'][i],dfNPV['Cumulative NPV'][i]), ha='center', va='bottom')

# Box plot
dfNPV = dfNPV.reset_index(drop=True)
idxLow = dfNPV.index[dfNPV['Hydrogen Price']=='Low']
dataLow = dfNPV.iloc[idxLow]['Cumulative NPV']
idxMedian = dfNPV.index[dfNPV['Hydrogen Price']=='Median']
dataMedian = dfNPV.iloc[idxMedian]['Cumulative NPV']
idxHigh = dfNPV.index[dfNPV['Hydrogen Price']=='High']
dataHigh = dfNPV.iloc[idxHigh]['Cumulative NPV']

fig, ax = plt.subplots()
sns.boxplot([dataLow, dataMedian, dataHigh],
palette='Blues').set(title='Cumulative NPV distribution for the three hydrogen market scenarios')
ax.set_xticklabels(['Low hydrogen price', 'Baseline hydrogen price', 'High hydrogen price'], rotation=0, fontsize=8)
ax.axes.grid(True)
ax.axes.set_xlabel('Hydrogen market scenario')
ax.axes.set_ylabel('Cumulative NPV [MCAD]')
ticks = ax.get_yticks()/10**6 # Conversion en MCAD
ax.set_yticklabels(ticks) # Conversion en MCAD
ax.yaxis.offsetText.set_fontsize(12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.figure()
columns = ['Low methanol', 'Baseline methanol', 'High methanol']
rows = ['Low electricity', 'Baseline electricity', 'High electricity']
LowH2NPV = [[dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eLmLh2L']]['Cumulative NPV'].iloc[0], \
    dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eLmMh2L']]['Cumulative NPV'].iloc[0], \
        dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eLmHh2L']]['Cumulative NPV'].iloc[0]], \
            [dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eMmLh2L']]['Cumulative NPV'].iloc[0], \
                dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eMmMh2L']]['Cumulative NPV'].iloc[0], \
                    dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eMmHh2L']]['Cumulative NPV'].iloc[0]], \
                        [dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eHmLh2L']]['Cumulative NPV'].iloc[0], \
                            dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eHmMh2L']]['Cumulative NPV'].iloc[0], \
                                dfNPV.iloc[dfNPV.index[dfNPV['SeriesOpti']=='eHmHh2L']]['Cumulative NPV'].iloc[0]]]

dfNPVLowH2 = pd.DataFrame(LowH2NPV, columns=columns, index=rows)/10**6 # Conversion en MCAD
 
ax = sns.heatmap(dfNPVLowH2, annot=True, cmap='Blues', cbar_kws={'label': '[MCAD]'}).set(title='Cumulative NPV distribution in low hydrogen price market scenario')
#ax.yaxis.offsetText.set_fontsize(12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

# Box plot IRR
dfNPV = dfNPV.reset_index(drop=True)
dataLow = dfNPV.iloc[idxLow]['IRR']
dataMedian = dfNPV.iloc[idxMedian]['IRR']
dataHigh = dfNPV.iloc[idxHigh]['IRR']

fig, ax = plt.subplots()
sns.boxplot([dataLow*100, dataMedian*100, dataHigh*100],
palette='Blues').set(title='IRR for the three hydrogen market scenarios')
ax.set_xticklabels(['Low hydrogen price', 'Baseline hydrogen price', 'High hydrogen price'], rotation=0, fontsize=8)
ax.axes.grid(True)
ax.axes.set_xlabel('Hydrogen market scenario')
ax.axes.set_ylabel('IRR [%]')
ax.yaxis.offsetText.set_fontsize(12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.show()

d =3
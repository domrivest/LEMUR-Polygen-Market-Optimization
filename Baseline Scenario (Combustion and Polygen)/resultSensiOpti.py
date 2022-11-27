""" Results of sensitvity analysis plotted with matplotlib
Author : Dominic Rivest
Highest nominal power in baseline market scenario, prices varying by ± 15 %
Created in 2022 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seriesOpti = ['BiomassInc15(8000kW)', 'BiomassDec15(8000kW)', 'ElecInc15(8000kW)', 'ElecDec15(8000kW)', 'MeOHInc15(8000kW)', \
    'MeOHDec15(8000kW)', 'H2Inc15(8000kW)', 'H2Dec15(8000kW)'] # Sensitivity analysis per serie change (and related name)

# Initial values and parameters

years = range(2022, 2037) # 15 years Equipment life (2037 not included)
discountRate = 0.10 # 10 % base discount rate
OM = 0.055 # 5.5% Of FIC for operation and maintenance
showPLT = 1 # 1 show matplotlib graphs
tau = 0.25 # 15 minute timesteps
TRNSYSInfo = pd.read_excel('Serre_plant-condensation-control_building.xlsx')  # Greenhouse heat demand
heatDemandIni = TRNSYSInfo['MW pour 20 000 m2']*1000*tau # Conversion from MW to kW (mettre *tau pour kWh)
nominalPower = np.multiply(range(1600, 8001, 1600), tau) # System nominal power (kW*15minutes, kWh)
inflationArray = np.ones(15)*1.0304 # Canadian mean inflation rate 2017-2021
inflationArray[0] = 1.0684 # Adjusted for 2022 Jan-Sep mean inflation value for first year
refNPV = 17571384.62622283 # From resultsTreatment (baseline market scenario for electricity, methanol and hydrogen)
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
    temp_df = pd.read_csv("CSV/sensiOpti/optimalValues"+seriesOpti[i]+".csv")
    temp_df['Heat demand (kWh)'] = heatDemandIni[0:len(heatDemandIni)-1]
    temp_df['Biomass heat (kWh)'] = temp_df['Biomass']*temp_df['operatingMode1']*etaSys['Heat'][1]+temp_df['Biomass']*temp_df['operatingMode2']*etaSys['Heat'][2]\
        +temp_df['Biomass']*temp_df['operatingMode3']*etaSys['Heat'][3]+temp_df['Biomass']*temp_df['operatingMode4']*etaSys['Heat'][4]
    df_optimalValues.append(temp_df)
    temp_dfT = pd.read_csv("CSV/sensiOpti/optimalValuesTotal"+seriesOpti[i]+".csv")
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

dfNPV['sensiVariable'] = ['Biomass', 'Electricity', 'MeOH', 'H²']
dfNPV['lowerNPV'] = -np.array([df_polygenCashFlow.loc[2036, 'Cumulative NPVBiomassDec15(8000kW)'], df_polygenCashFlow.loc[2036, 'Cumulative NPVElecInc15(8000kW)'], \
    df_polygenCashFlow.loc[2036, 'Cumulative NPVMeOHInc15(8000kW)'], df_polygenCashFlow.loc[2036, 'Cumulative NPVH2Inc15(8000kW)']])
dfNPV['upperNPV'] = -np.array([df_polygenCashFlow.loc[2036, 'Cumulative NPVBiomassInc15(8000kW)'], df_polygenCashFlow.loc[2036, 'Cumulative NPVElecDec15(8000kW)'], \
    df_polygenCashFlow.loc[2036, 'Cumulative NPVMeOHDec15(8000kW)'], df_polygenCashFlow.loc[2036, 'Cumulative NPVH2Dec15(8000kW)']])
dfNPV['lowerChange'] = 1-dfNPV['lowerNPV']/refNPV
dfNPV['upperChange'] = dfNPV['upperNPV']/refNPV-1

# Graphique sensibilité NPV

dfNPV = dfNPV.sort_values(by=['lowerNPV'], ascending=True)
index = dfNPV.index
column0 = dfNPV['lowerNPV']
column1 = dfNPV['upperNPV']

fig, axes = plt.subplots()

axes.barh(dfNPV['sensiVariable'], column1, color='white', zorder=2)
axes.barh(dfNPV['sensiVariable'], column0, color='lightseagreen', zorder=1)
axes.axvline(refNPV, ls='--', zorder=3)
axes.set_title('NPV sensitivity range for the major resources')
axes.set_xlabel('NPV [CAD]')

axes.set_xlim([10000000, 26000000])

plt.show()


d =3
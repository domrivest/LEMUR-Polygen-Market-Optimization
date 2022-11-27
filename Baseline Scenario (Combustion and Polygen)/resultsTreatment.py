""" Analysis and plots of the optimisation results using plotly and matplotlib
The Matplotlib graph are used in the paper, the plotly ones are used for data exploration
Author : Dominic Rivest
Created in 2022 """

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
from datetime import datetime, timedelta


optiFileNames = ['400.0(1600kW)', '800.0(3200kW)', '1200.0(4800kW)', '1600.0(6400kW)', '2000.0(8000kW)']
optiFileNamesCombustion = ['325.0(1300kW)', '650.0(2600kW)', '975.0(3900kW)', '1300.0(5200kW)', '1625.0(6500kW)']

# Initial values and parameters

CO2costs = 0.2*380*1000 # Annual CO2 consumption costs for combustion system (0.2 $/kg * 380 tons)
years = range(2022, 2037) # 15 years Equipment life (2037 not included)
discountRate = 0.10 # 10 % base discount rate
OM = 0.055 # 5.5% Of FIC for operation and maintenance
showPLT = 1 # 1 show matplotlib graphs
showPlotly = 0 # 1 show Plotly graphs
tau = 0.25 # 15 minute timesteps
TRNSYSInfo = pd.read_excel('Serre_plant-condensation-control_building.xlsx')  # Greenhouse heat demand
heatDemandIni = TRNSYSInfo['MW pour 20 000 m2']*1000*tau # Conversion from MW to kW (mettre *tau pour kWh)
nominalPower = np.multiply(range(1600, 8001, 1600), tau) # System nominal power (kW*15minutes, kWh)
discountRateVector = range(0, 21, 1) # Vector to test discount rate impact on NPV
inflationArray = np.ones(15)*1.0304 # Canadian mean inflation rate 2017-2021
inflationArray[0] = 1.0684 # Adjusted for 2022 Jan-Sep mean inflation value for first year

for i in range(len(inflationArray)): # Cumulative inflation calculation
    if i > 0:
        inflationArray[i] = inflationArray[i-1]*inflationArray[i]

sumHeatDemand = sum(heatDemandIni)# Demande totale en chaleur pour l'année [kWh]
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
df_optimalValuesCombustion = []
df_optimalValuesTotal = []
 
# append datasets into the list, sum operating modes, calculate heat excess
for i in range(len(optiFileNames)):
    temp_df = pd.read_csv("CSV/optimalValues"+optiFileNames[i]+".csv")
    temp_df['Heat demand (kWh)'] = heatDemandIni[0:len(heatDemandIni)-1]
    temp_df['Biomass heat (kWh)'] = temp_df['Biomass']*temp_df['operatingMode1']*etaSys['Heat'][1]+temp_df['Biomass']*temp_df['operatingMode2']*etaSys['Heat'][2]\
        +temp_df['Biomass']*temp_df['operatingMode3']*etaSys['Heat'][3]+temp_df['Biomass']*temp_df['operatingMode4']*etaSys['Heat'][4]
    df_optimalValues.append(temp_df)
    temp_dfT = pd.read_csv("CSV/optimalValuesTotal"+optiFileNames[i]+".csv")
    temp_dfT['Max electricity'] = sum(temp_df['operatingMode1'])/(len(heatDemandIni)-1)
    temp_dfT['Max MeOH'] = sum(temp_df['operatingMode2'])/(len(heatDemandIni)-1)
    temp_dfT['Max H2'] = sum(temp_df['operatingMode3'])/(len(heatDemandIni)-1)
    temp_dfT['Max heat'] = sum(temp_df['operatingMode4'])/(len(heatDemandIni)-1)
    temp_dfT['Biomass heat (kWh)'] = sum(temp_df['Biomass heat (kWh)'])
    temp_dfT['Excess heat (kWh)'] = sum(temp_df['Biomass heat (kWh)'][heatDemandIni[0:len(heatDemandIni)-1]<temp_df['Biomass heat (kWh)']]
    -heatDemandIni[0:len(heatDemandIni)-1][heatDemandIni[0:len(heatDemandIni)-1]<temp_df['Biomass heat (kWh)']])
    temp_dfT['Heat demand (kWh)'] = sum(temp_df['Heat demand (kWh)'])
    df_optimalValuesTotal.append(temp_dfT)
    temp_dfCombustion = pd.read_csv("CSV/optimalValuesCombustion"+optiFileNamesCombustion[i]+".csv")
    temp_dfCombustion['Heat demand (kWh)'] = heatDemandIni[0:len(heatDemandIni)-1]
    df_optimalValuesCombustion.append(temp_dfCombustion)

df_optimalValuesTotal = pd.concat(df_optimalValuesTotal[:], ignore_index=True) # Merge of the df in one

# Lecture des résultats de combustion

df_optimalValuesTotalCombustion = pd.read_csv("CSV/optimalValuesTotalCombustion.csv")

# Ajout et correction des valeurs de somme

df_optimalValuesTotal['nominalPower'] = ['1600', '3200', '4800', '6400', '8000']
df_optimalValuesTotalCombustion['nominalPower'] = ['1300', '2600', '3900', '5200', '6500']
df_optimalValuesTotalCombustion['ObjectiveFunction'] = df_optimalValuesTotalCombustion['ObjectiveFunction'] + CO2costs
df_optimalValuesTotalCombustion['BiomassCost'] = -df_optimalValuesTotalCombustion['BiomassCost']
df_optimalValuesTotalCombustion['PropaneCost'] = -df_optimalValuesTotalCombustion['PropaneCost']
df_optimalValuesTotalCombustion['CO2Cost'] = -CO2costs
df_optimalValuesTotal['BiomassCost'] = -df_optimalValuesTotal['BiomassCost']
df_optimalValuesTotal['MeOHCostIn'] = -df_optimalValuesTotal['MeOHCostIn']

# Lecture des fichiers de EC
df_polygenEC = pd.read_csv('CSV/polygenEC.csv')
df_combustionEC = pd.read_csv('CSV/combustionEC.csv')

# Annual cash flow and cumulative NPV calculations

# Ajouts des fixed OPEX à df_optimalValuesTotal
df_optimalValuesTotal['Fixed OPEX'] = -df_polygenEC['Total FIC']*OM
df_optimalValuesTotal['Annual cash flow sum'] = df_optimalValuesTotal.loc[:, ['ElecRevenue', 'MeOHRevenue', 'H2Revenue', 'BiomassCost', 'MeOHCostIn', 'Fixed OPEX']].sum(axis=1)
df_optimalValuesTotalCombustion['Fixed OPEX'] = -df_combustionEC['Total FIC']*OM
df_optimalValuesTotalCombustion['Annual cash flow sum'] = df_optimalValuesTotalCombustion.loc[:, ['BiomassCost', 'PropaneCost', 'CO2Cost', 'Fixed OPEX']].sum(axis=1)

def cashFlow(df_EC, df_optimalTotal, years, discountRate, OM, inflationArray):
    temp_df = pd.DataFrame()
    temp_df.index = years
    temp_df['Year'] = years
    for year in years:
        for j in range(0, len(df_optimalTotal['nominalPower'])):
            temp_df.loc[year, 'Cash Flow' + str(df_optimalTotal.loc[j, 'nominalPower'])] = (df_optimalTotal.loc[j, 'ObjectiveFunction']+OM*df_EC.loc[j, 'Total FIC'])*inflationArray[year-years[0]]/(1 + discountRate)**(year-years[0]+1)
            temp_df.loc[year, 'Cumulative NPV' + str(df_optimalTotal.loc[j, 'nominalPower'])] = df_EC.loc[j, 'Total FIC'] \
                +sum(temp_df.loc[years[0]:year, 'Cash Flow' + str(df_optimalTotal.loc[j, 'nominalPower'])])
    return temp_df

def getNPVPoly(df_EC, df_optimalTotal, discountRateVector, OM, systemPowerConfig, inflationArray):
    temp_df = pd.DataFrame()
    temp_df['Year'] = years
    tempSum_df = pd.DataFrame()
    tempSum_df['Discount Rate'] = discountRateVector
    for discountRate in discountRateVector : 
        for year in range(len(years)):
            temp_df.loc[year, 'OPEX'] = (OM*df_EC.loc[systemPowerConfig, 'Total FIC'] - df_optimalTotal.loc[systemPowerConfig, 'BiomassCost'] - df_optimalTotal.loc[systemPowerConfig, 'MeOHCostIn'])*inflationArray[year]/(1 + discountRate/100)**(year+1)
            temp_df.loc[year, 'Cash Flow'] = (df_optimalTotal.loc[systemPowerConfig, 'ObjectiveFunction'] + OM*df_EC.loc[systemPowerConfig, 'Total FIC'])*inflationArray[year]/(1 + discountRate/100)**(year+1)
            temp_df.loc[year, 'Cumulative NPV'] = df_EC.loc[systemPowerConfig, 'Total FIC'] \
                +sum(temp_df.loc[0:year, 'Cash Flow'])
        tempSum_df.loc[discountRate, 'Cumulative NPV over life'] = temp_df.loc[len(years)-1, 'Cumulative NPV']
        tempSum_df.loc[discountRate, 'LCOE'] = (sum(temp_df['OPEX'])+df_EC.loc[systemPowerConfig, 'Total FIC'])/(df_optimalTotal.loc[systemPowerConfig, 'H2']\
            +df_optimalTotal.loc[systemPowerConfig, 'Elec']+df_optimalTotal.loc[systemPowerConfig, 'MeOH']+df_optimalTotal.loc[systemPowerConfig, 'Heat']+\
                +df_optimalTotal.loc[systemPowerConfig, 'MeOHInput']+df_optimalTotal.loc[systemPowerConfig, 'Excess heat (kWh)'])
    return tempSum_df

def getNPVCombustion(df_EC, df_optimalTotal, discountRateVector, OM, systemPowerConfig, inflationArray):
    temp_df = pd.DataFrame()
    temp_df['Year'] = years
    tempSum_df = pd.DataFrame()
    tempSum_df['Discount Rate'] = discountRateVector
    for discountRate in discountRateVector : 
        for year in range(len(years)):
            temp_df.loc[year, 'OPEX'] = (OM*df_EC.loc[systemPowerConfig, 'Total FIC'] + df_optimalTotal.loc[systemPowerConfig, 'BiomassCost'] + df_optimalTotal.loc[systemPowerConfig, 'PropaneCost'])*inflationArray[year]/(1 + discountRate/100)**(year+1)
            temp_df.loc[year, 'Cash Flow'] = (df_optimalTotal.loc[systemPowerConfig, 'ObjectiveFunction'] + OM*df_EC.loc[systemPowerConfig, 'Total FIC'])*inflationArray[year]/(1 + discountRate/100)**(year+1)
            temp_df.loc[year, 'Cumulative NPV'] = df_EC.loc[systemPowerConfig, 'Total FIC'] \
                +sum(temp_df.loc[0:year, 'Cash Flow'])
        tempSum_df.loc[discountRate, 'Cumulative NPV over life'] = temp_df.loc[len(years)-1, 'Cumulative NPV']
        tempSum_df.loc[discountRate, 'LCOH'] = (sum(temp_df['OPEX'])+df_EC.loc[systemPowerConfig, 'Total FIC'])/df_optimalTotal.loc[systemPowerConfig, 'Heat']
    return tempSum_df



df_polygenCashFlow = cashFlow(df_polygenEC, df_optimalValuesTotal, years, discountRate, OM, inflationArray)
df_combustionCashFlow = cashFlow(df_combustionEC, df_optimalValuesTotalCombustion, years, discountRate, OM, inflationArray)
df_polygenNPV = getNPVPoly(df_polygenEC, df_optimalValuesTotal, discountRateVector, OM, 4, inflationArray) # 8000 kW
df_combustionNPV = getNPVCombustion(df_combustionEC, df_optimalValuesTotalCombustion, discountRateVector, OM, 3, inflationArray) # 5200 kW

d =3

# Plot section

times = []
ts = datetime(2019, 1, 1, 0, 0, 0)
times = np.array([ts + timedelta(minutes=15*i) for i in range(0, 35041)])

# matPlotlib section
colorMapOp = ['#ffed6f', '#80b1d3', '#8dd3c7', '#d9d9d9', 'forestgreen', 'coral'] # electricity, MeOH, hydrogen, Heat, biomass, MeOH input
colorMapCashFlow = ['#ffed6f', '#80b1d3', '#8dd3c7', 'forestgreen', 'coral', 'gainsboro'] # electricity, MeOH, hydrogen, Heat, biomass, MeOH input, operationMaintenance (Fixed OPEX)
colorMapCashFlowCombustion = ['forestgreen', 'bisque', 'darkslategray', 'gainsboro']  # Biomass, propane, CO2, OPEX

if showPLT == 1:
    
    # Annual heat demand from Trynsys
    plt.figure()
    plt.plot(times, heatDemandIni/tau, color="darkorchid", linewidth=0.5)
    plt.xlabel("Time")
    plt.ylabel("Heat demand [kW]")
    plt.title("Greenhouse heat demand from TRNSYS model")
    ax = plt.gca()
    # formatters' options
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(DateFormatter('%b'))

    # Relative occurence of operating modes
    ax = df_optimalValuesTotal.plot("nominalPower", ["Max electricity", "Max MeOH", "Max H2", "Max heat"], 'bar',
    stacked=True,
    xlabel = 'Nominal polygeneration system power [kW]',
    ylabel = 'Operating mode relative distribution [-]',
    title = 'Operating mode relative distribution against system power in the baseline market scenario',
    color = colorMapOp
    )
    ax.legend(['Maximum electricity', 'Maximum MeOH', 'Maximum H2', 'Maximum heat'])

    # Annual cash flows (polygen)
    ax = df_optimalValuesTotal.plot("nominalPower", ['ElecRevenue', 'MeOHRevenue', 'H2Revenue', 'BiomassCost', 'MeOHCostIn', 'Fixed OPEX'], 'bar',
    stacked=True,
    xlabel = 'Nominal polygeneration system power [kW]',
    ylabel = 'Cash flow [CAD]',
    title = 'Annual polygeneration system cash flows against \n system power in the baseline market scenario (2022)',
    color = colorMapCashFlow,
    grid = True
    )
    ax.bar(df_optimalValuesTotal['nominalPower'], df_optimalValuesTotal['Annual cash flow sum'], color='black', width=0.25)
    ax.legend(['Electricity', 'MeOH', 'Hydrogen', 'Biomass', 'MeOH (peaking burner)', 'Fixed OPEX', 'Annual cash flow sum'])
    # Ajout des valeurs en bout de barre
    for i in range(len(df_optimalValuesTotal['nominalPower'])):
        if df_optimalValuesTotal['Annual cash flow sum'][i] <= 0:
            ax.annotate(str(int(round(df_optimalValuesTotal['Annual cash flow sum'][i], -4))), xy=(df_optimalValuesTotal['nominalPower'][i],df_optimalValuesTotal['Annual cash flow sum'][i]), ha='center', va='top')
        else:
            ax.annotate(str(int(round(df_optimalValuesTotal['Annual cash flow sum'][i], -4))), xy=(df_optimalValuesTotal['nominalPower'][i],df_optimalValuesTotal['Annual cash flow sum'][i]), ha='center', va='bottom')

    # Annual cash flows (combustion)
    ax = df_optimalValuesTotalCombustion.plot("nominalPower", ['BiomassCost', 'PropaneCost', 'CO2Cost', 'Fixed OPEX'], 'bar',
    stacked=True,
    xlabel = 'Nominal combustion system power [kW]',
    ylabel = 'Cash flow [CAD]',
    title = 'Annual combustion-based system cash flows \n against system power (2022)',
    color = colorMapCashFlowCombustion,
    grid = True
    )
    ax.bar(df_optimalValuesTotalCombustion['nominalPower'], df_optimalValuesTotalCombustion['Annual cash flow sum'], color='black', width=0.25)
    ax.legend(['Biomass', 'Propane (peaking burner)', 'CO2', 'Fixed OPEX', 'Annual cash flow sum'])
    # Ajout des valeurs en bout de barre
    for i in range(len(df_optimalValuesTotalCombustion['nominalPower'])):
        ax.annotate(str(int(round(df_optimalValuesTotalCombustion['Annual cash flow sum'][i], -4))), xy=(df_optimalValuesTotalCombustion['nominalPower'][i],df_optimalValuesTotalCombustion['Annual cash flow sum'][i]-5000), ha='center', va='top')

    # Polygeneration 8000 kW system operation (MeOHIn, Biomass, Heat demand)
    
    ax = df_optimalValues[4].plot(
        'Unnamed: 0',
        ["Heat demand (kWh)", "Biomass", "MeOHInput", 'MeOH', 'H2', 'Elec'],
        xlabel='Time steps [15 minutes]',
        ylabel='Energy [kWh]',
        color=['darkslategray', 'forestgreen', 'coral', '#80b1d3', '#8dd3c7', '#ffed6f'], # electricity, MeOH, hydrogen, Heat, biomass, MeOH input, operationMaintenance (Fixed OPEX)
        title="Annual greenhouse heat demand and energy flows for the 8000 kW polygeneration system"
        )
    ax.legend([
                'Greenhouse heat demand',
                'Biomass consumption',
                'Methanol consumption',
                'Methanol production',
                'Hydrogen production',
                'Electricity production'
        ])

    # Biomass boiler 5200 kW system operation (Propane, Biomass, Heat demand)
    
    ax = df_optimalValuesCombustion[3].plot(
        'Unnamed: 0',
        ["Heat demand (kWh)", "Biomass", "Propane"],
        xlabel='Time steps [15 minutes]',
        ylabel='Energy [kWh]',
        color=['darkslategray', 'forestgreen', 'bisque'],
        title="Annual greenhouse heat demand and biomass consumption for the 5200 kW boiler system"
        )
    ax.legend([
                'Greenhouse heat demand',
                'Biomass consumption',
                'Propane consumption',
        ])

    # Polygeneration system NPV for every system size
    df_polygenCashFlow['Cumulative NPV1600'] = -df_polygenCashFlow['Cumulative NPV1600']
    df_polygenCashFlow['Cumulative NPV3200'] = -df_polygenCashFlow['Cumulative NPV3200']
    df_polygenCashFlow['Cumulative NPV4800'] = -df_polygenCashFlow['Cumulative NPV4800']
    df_polygenCashFlow['Cumulative NPV6400'] = -df_polygenCashFlow['Cumulative NPV6400']
    df_polygenCashFlow['Cumulative NPV8000'] = -df_polygenCashFlow['Cumulative NPV8000']
    blues = plt.cm.Blues(np.linspace(0.3, 1, 5))

    ax = df_polygenCashFlow.plot(
        'Year',
        ['Cumulative NPV1600', 'Cumulative NPV3200', 'Cumulative NPV4800', 'Cumulative NPV6400', 'Cumulative NPV8000'],
        'bar',
        color = blues,
        xlabel='Year',
        ylabel='Cumulative NPV [CAD]',
        title='Cumulative NPV system life - Polygeneration',
        grid=True
    )
    
    ax.ticklabel_format(style='sci', axis='y')
    ax.legend([
        '1600 kW',
        '3200 kW',
        '4800 kW',
        '6400 kW',
        '8000 kW'
        ])

    # Combustion system NPV for every system size

    df_combustionCashFlow['Cumulative NPV1300'] = -df_combustionCashFlow['Cumulative NPV1300']
    df_combustionCashFlow['Cumulative NPV2600'] = -df_combustionCashFlow['Cumulative NPV2600']
    df_combustionCashFlow['Cumulative NPV3900'] = -df_combustionCashFlow['Cumulative NPV3900']
    df_combustionCashFlow['Cumulative NPV5200'] = -df_combustionCashFlow['Cumulative NPV5200']
    df_combustionCashFlow['Cumulative NPV6500'] = -df_combustionCashFlow['Cumulative NPV6500']
    blues = plt.cm.Oranges(np.linspace(0.3, 1, 5))

    ax = df_combustionCashFlow.plot(
        'Year',
        ['Cumulative NPV1300', 'Cumulative NPV2600', 'Cumulative NPV3900', 'Cumulative NPV5200', 'Cumulative NPV6500'],
        'bar',
        color = blues,
        xlabel='Year',
        ylabel='Cumulative NPV [CAD]',
        title='Cumulative NPV through system life - Biomass boiler',
        grid=True
    )

    ax.legend([
        '1300 kW',
        '2600 kW',
        '3900 kW',
        '5200 kW',
        '6500 kW'
        ])
    
    # Cumulative NPV vs discount rate

    df_NPV = pd.DataFrame()
    df_NPV['Discount Rate'] = discountRateVector
    df_NPV['Cumulative NPV for 5200 kW combustion'] = -df_combustionNPV['Cumulative NPV over life']
    df_NPV['Cumulative NPV for 8000 kW polygen'] = -df_polygenNPV['Cumulative NPV over life']
    ax = df_NPV.plot("Discount Rate", ["Cumulative NPV for 8000 kW polygen", "Cumulative NPV for 5200 kW combustion"],
    'bar',
    stacked=True,
    color=['cornflowerblue', 'darkorange'],
    xlabel='Discount rate [%]',
    ylabel='Cumulative NPV [CAD]',
    grid=True,
    title="Cumlative NPV against discount rate"
    )
    ax.legend(['8000 kW polygeneration system', '5200 kW combustion-based system'])

    
    plt.show()

# Plotly section

if showPlotly == 1:
    figOp = []
    figTimePolygen = []
    figTimeCombustion = []
    figNpv = []

    # Graphique des flux d'énergie selon la puissance

    figOp.append(px.bar(df_optimalValuesTotal, x="nominalPower", y=["Biomass", "MeOHInput", "Elec", "MeOH", "H2", "Heat demand (kWh)"],
                barmode='group',
                labels={
                        "nominalPower": "System nominal power (kW)",
                        "value": "Energy (kWh)"
                    },
                    title="System inputs and outputs vs system nominal power"
                    )
                    )

    # Graphique de la proportion de chaque mode en fonction de la puissance du système

    figOp.append(px.bar(df_optimalValuesTotal, x="nominalPower", y=["Max electricity", "Max MeOH", "Max H2", "Max heat"],
                barmode='group',
                labels={
                        "nominalPower": "System nominal power (kW)",
                        "value": "Relative occurence"
                    },
                    title="Relative occurence of operation modes vs system nominal power"
                    )
                    )

    # Graphique des différents flux monétaire en fonction de la puissance du système

    figOp.append(px.bar(df_optimalValuesTotal, x="nominalPower", y=['BiomassCost', 'MeOHCostIn', 'ElecRevenue', 'MeOHRevenue', 'H2Revenue'],
                barmode='relative',
                labels={
                        "nominalPower": "System nominal power (kW)",
                        "value": "Cash flow (CAD)"
                    },
                    title="Annual cash flow breakdown vs system nominal power"
                )
    )



    # Ajout de la somme des flux

    figOp[len(figOp)-1].add_bar(
            x=df_optimalValuesTotal['nominalPower'],
            y=-df_optimalValuesTotal['ObjectiveFunction'],
            name = 'Cash flow sum',
            base = 0,
            width = 0.2,
            marker_color="black"
        )

    figOp[len(figOp)-1].update_xaxes(type='category')
    figOp[len(figOp)-1].update_xaxes(title_text='System nominal power (kW)')
    figOp[len(figOp)-1].update_yaxes(title_text='Cash flow (CAD)')

    # Graphiques des pertes en chaleur

    figOp.append(px.bar(df_optimalValuesTotal, x="nominalPower", y='Excess heat (kWh)',
                labels={
                        "nominalPower": "System nominal power (kW)",
                        "value" : "Excess heat (kWh)"
                    },
                    title="Total excess heat vs system nominal power"
                )
    )
    figOp[len(figOp)-1].update_xaxes(type='category')

    # Line plot of biomass heat, meoh-input heat and heat demand

    for i in range(0,len(nominalPower)):
        figTimePolygen.append(
            px.line(
                df_optimalValues[i],
                x = df_optimalValues[i].index,
                y = ["Heat demand (kWh)", "Biomass", "MeOHInput"],
                labels={
                        "nominalPower": "System nominal power (kW)",
                        "value" : "Energy"
                    },
                    title="Heat demand, Biomass energy input and MeOH energy input vs time for nominal power = " + str(nominalPower[i]/tau) + "kW"
            )
        )

    # Bar chart of operation costs of biomass-propane burners

    figOp.append(px.bar(df_optimalValuesTotalCombustion, x="nominalPower", y=['BiomassCost', 'PropaneCost', 'ObjectiveFunction'],
                barmode='group',
                labels={
                        "nominalPower": "Combustion system biomass nominal power (kW)",
                        "value": "Cash flow (CAD)"
                    },
                    title="Annual cash flow breakdown vs combustion system nominal power"
                )
    )

    # Line chart des consommations de biomasse et de propane avec la demande en chaleur

    for i in range(0,len(df_optimalValuesTotalCombustion['nominalPower'])):
        figTimeCombustion.append(
            px.line(
                df_optimalValuesCombustion[i],
                x = df_optimalValuesCombustion[i].index,
                y = ["Heat demand (kWh)", "Biomass", "Propane"],
                labels={
                        "nominalPower": "System nominal power (kW)",
                        "value" : "Energy"
                    },
                    title="Heat demand, Biomass and propane energy input vs time for nominal power = " + df_optimalValuesTotalCombustion['nominalPower'][i] + "kW"
            )
        )

    # Graphiques des discounted cash flow

    figNpv.append(px.bar(df_polygenCashFlow, x="Year", y=["Cash Flow1600", "Cash Flow3200", "Cash Flow4800", "Cash Flow6400", "Cash Flow8000"],
                barmode='group',
                labels={
                        "value": "Annual cash flow (CAD 2022)"
                    },
                    title="Discounted cash flow over life of polygeneration system"
                    )
                    )
    
    figNpv.append(px.bar(df_combustionCashFlow, x="Year", y=["Cash Flow1300", "Cash Flow2600", "Cash Flow3900", "Cash Flow5200", "Cash Flow6500"],
                barmode='group',
                labels={
                        "value": "Annual cash flow (CAD 2022)"
                    },
                    title="Discounted cash flow over life of combustion system"
                    )
                    )

    # Graphiques des discounted NPV cumulatifs

    figNpv.append(px.bar(df_polygenCashFlow, x="Year", y=["Cumulative NPV1600", "Cumulative NPV3200", "Cumulative NPV4800", "Cumulative NPV6400", "Cumulative NPV8000"],
                barmode='group',
                labels={
                        "value": "Cumulative NPV (CAD 2022)"
                    },
                    title="Cumlative cash flow over life of polygeneration system"
                    )
                    )

    figNpv.append(px.bar(df_combustionCashFlow, x="Year", y=["Cumulative NPV1300", "Cumulative NPV2600", "Cumulative NPV3900", "Cumulative NPV5200", "Cumulative NPV6500"],
                barmode='group',
                labels={
                        "value": "Cumulative NPV (CAD 2022)"
                    },
                    title="Cumlative cash flow over life of combustion system"
                    )
                    )

    df_NPV = pd.DataFrame()
    df_NPV['Discount Rate'] = discountRateVector
    df_NPV['Cumulative NPV for 5200 kW combustion'] = df_combustionNPV['Cumulative NPV over life']
    df_NPV['Cumulative NPV for 8000 kW polygen'] = df_polygenNPV['Cumulative NPV over life']
    figNpv.append(px.bar(df_NPV, x="Discount Rate", y=["Cumulative NPV for 8000 kW polygen", "Cumulative NPV for 5200 kW combustion"],
                barmode='group',
                labels={
                        "value": "Cumulative NPV (CAD 2022)"
                    },
                    title="Cumlative NPV vs discount rate"
                    )
                    )


    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        # represents the browser address bar and doesn't render anything
        dcc.Location(id='url', refresh=False),

        dcc.Link('Navigate to "/"', href='/'),
        html.Br(),
        dcc.Link('Navigate to "/operation"', href='/operation'),
        html.Br(),
        dcc.Link('Navigate to "/timePolygen"', href='/timePolygen'),
        html.Br(),
        dcc.Link('Navigate to "/timeCombustion"', href='/timeCombustion'),
        html.Br(),
        dcc.Link('Navigate to "/npv"', href='/npv'),

        # content will be rendered in this element
        html.Div(id='page-content')
    ])

    @app.callback(Output('page-content', 'children'),
                [Input('url', 'pathname')])
    def display_page(pathname):
        content = []
        if pathname == "/operation":
            content = []
            for j in range(0, len(figOp)) :
                content.append(
                    html.Div([
                        dcc.Graph(
                            id='graph'+str(j),
                            figure=figOp[j]
                            ),
                    ])
                )
            return content
        elif pathname == "/timePolygen":
            content = []
            for j in range(0, len(figTimePolygen)) :
                content.append(
                    html.Div([
                        dcc.Graph(
                            id='graph'+str(j),
                                figure=figTimePolygen[j]
                            ),
                    ])
                )
            return content
        elif pathname == "/timeCombustion":
            content = []
            for j in range(0, len(figTimeCombustion)) :
                content.append(
                    html.Div([
                        dcc.Graph(
                            id='graph'+str(j),
                                figure=figTimeCombustion[j]
                            ),
                    ])
                )
            return content
        elif pathname == "/npv":
            content = []
            for j in range(0, len(figNpv)) :
                    content.append(
                        html.Div([
                            dcc.Graph(
                                id='graph'+str(j),
                                figure=figNpv[j]
                                ),
                    ])
                )
            return content
        else:
            return html.Div([
                html.H3(f'Bonsoir, vous consultez la page {pathname}, sélectionnez une page pour voir les graphiques selon leur type'),
            ])
        


    if __name__ == '__main__':
        app.run_server(debug=True)

d = 66

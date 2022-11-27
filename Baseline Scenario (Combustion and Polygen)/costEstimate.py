""" Cost estimation of the subsystems of the polygeneration and combustion systems
Author : Dominic Rivest
Created in 2022 """

import math
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# Input parameters

polygenNominalPower = range(1600, 8001, 1600) # Biomass nominal power of the polygeneration system (kW)
peakMeOHBurnerPower = {1600:5000, 3200:4000, 4800:2500, 6400:1500, 8000:1000} # Basé sur le balance du heat demand en mode max heat arrondie au 500 kW en haut le plus près (et pour le maintenance break)
combustionNominalPower = range(1300, 1300*5+1, 1300) # Biomass nominal power of the combustion system (kW)
TRNSYSInfo = pd.read_excel('Serre_plant-condensation-control_building.xlsx')  # Greenhouse heat demand
heatDemandIni = TRNSYSInfo['MW pour 20 000 m2']*1000 # Conversion from MW to kW (mettre *tau pour kWh)
peakPowerDemand = max(heatDemandIni)
txFrIn = 1.20 # Taxes + freight + insurance factor - Rules of Thumb
hmOfFiEx = 1.40 # Indirects for home office and field expenses
ctrFee = 1.05 # Contractor fee - Rules of Thumb
ctngcy = 1.15 # Contingency for delays - Rules of Thumb
ctngncySc = 1.1 # Contingency for change in scope - Rules of Thumb


# Conversion factors and additional information

from iapws import IAPWS97
woodOdmtHhv = 21/3.6*10**3 # (kWh/odmt) 21 MJ/kg to kWh/ODMT
wood45Hhv = 14/3.6*10**3 # kWh/t moist 14 mJ/kg 
USD2CAD = 1.27
EUR2CAD = 1.35
SEK2CAD = 0.13
cepciCurrent = 797.6
evalYear = 2022
# Definition of equipment Cost scaling function

def ecScaling(refCost, refCapacity, refCepci, currencyConv, cepciCurrent, n, capacity):
    cost = refCost*currencyConv*(capacity/refCapacity)**n*cepciCurrent/refCepci
    return cost

# Gasifier EC (Mean of two references)

gasRef1Cost = 29490000 # SEK 2014 GoBiGas
gasRef1Cepci = 576.1 # Cepci 2014
gasRef1Capacity = 6 # tons/h odmt biomass

gasScalingFactor = 0.70 # Comes from gobigas article

gasRef2Cost = 8000000/5.03 # Euro 2003 (divided to find base cost)
gasRef2Cepci = 402.0 # Cepci 2003
gasRef2Capacity = 2800 # kW fuel input

# Cyclone + cyclone fan EC (Rules of Thumb)

tempCyclone = 800
cycloneRefCost = 35000 # USD for cepci=1000
cycloneRefCepci = 1000 # Cepci from rules of thumb
cycloneRefCapacity = 10 # m3/s
cycloneScalingFactor = 0.56
cycloneRefLM = 2.5 # L+M* factor from Rules of Thumb
cycloneFanRefCost = 27750 # USD for cepci=1000
cycloneFanRefCepci = 1000 # Cepci from rules of Thumb
cycloneFanRefCapacity = 10 # m3/s
cycloneFanScalingFactor = 0.93
cycloneFanRefLM = 1.7 # L+M* factor from Rules of Thumb

# Nickel fluidized bed EC (Rules of Thumb)

tempNi = 780+273.15 # Nickel fluidized bed Temp (k)
pressureNi = 0.1013 # Bed bressure (MPa)
steamNi = IAPWS97(P=pressureNi, T=tempNi) # (m3/kg)
steamVNi = steamNi.v
NiFluidizedBedRefCost = 4200000 # USD for cepci=1000
NiFluidizedBedRefCepci = 1000 # Cepci from Rules of Thumb
NiFluidizedBedRefCapacity = 12 # fluidized bed volume (m3)
NiFluidizedBedScalingFactor = 0.67
NiFluidizedBedRefTM = 2 # TM factor from Rules of Thumb - Total module

# ZnO guard bed EC (A standardized methodology for the techno-economic evaluation of alternative fuels – A case study)
tempZnO = 400+273.15 # ZnO fluidized bed Temp (k)
pressureZnO = 0.1013 # Bed bressure (MPa)
steamZnO = IAPWS97(P=pressureZnO, T=tempZnO) # Properties
steamVZnO = steamZnO.v # (m3/kg)
ZnOGuardBedRefCost = 0.02*1000000 # EUR 2014 - FOB Cost
ZnOGuardBedRefCepci = 576.1 # Cepci 2014
ZnOGuardBedRefCapacity = 8 # m3/s
ZnOGuardBedScalingFactor = 1 
ZnOGuardBedRefLM = 2.3 # Fixed bed w/ catalyst L+M* factor from Rules of Thumb
ZnOGuardBedRefIns = ecScaling(63000, 1, 1000, USD2CAD, cepciCurrent, 1, 1) # 63000 USD for gas phase reactors (instrumentation cost)

# Flash separator EC

flashLiquidProp = IAPWS97(P=0.1013, T=40+273.15) # Properties
flashLiquidDensity = flashLiquidProp.rho # kg/m3
flashGasDensity = 0.95 # kg/m3 (Co-gasification of coal dans hardwood pellets : A case study)
flashMaxVelocity = 0.13*((flashLiquidDensity-flashGasDensity)/flashGasDensity)**0.5*0.675 # rules of thumb in chemical engineering knockout pot (safety factor 67,5%)
flashRefCost = 100000 # FOB cost USD for cepci=1000
flashRefCepci = 1000 # Cepci from Rules of Thumb
flashRefCapacity = 20 # Volume [m3]
flashScalingFactor = 0.52
flashRefLM = 2.3 # L+M* factor from Rules of Thumb
flashRefIns = ecScaling(8300, 1, 1000, USD2CAD, cepciCurrent, 1, 1) # 8300 USD (intermediate process tank ins cost)

# MeOH synthesis subsystem EC (w/ compressor)
compressorRefCost = 435000 # Rules of Thumb under 7 MPa
compressorRefCepci = 1000
compressorRefCapacity = 300 # kW
compressorScalingFactor = 0.85 # Rules of Thumb
compressorRefLM = 2.15 # L+M* factor from Rules of Thumb
compressorRefIns = 7000 # USD (compressor stage ins cost)
meohRefCost = 350000 # USD for CEPCI=1000, Rules of Thumb Section 6.12 PFTR: Multitube Fixed Bed Catalyst or Bed of Solid Inerts: Nonadiabatic
meohRefCepci = 1000 # From Rules of Thumb
meohRefCapacity = 3 # m3 internal volume
meohScalingFactor = 0.68
meohRefLM = 2.8 # L+M* factor from Rules of Thumb
meohRefIns = ecScaling(63000, 1, 1000, USD2CAD, cepciCurrent, 1, 1) # 63000 USD for gas phase reactors (instrumentation cost)

# SOFC EC

sofcRefCost = 983 # USD 2011 / kWe (Previously 356 USD 2018)
sofcRefCepci = 585.7 # Cepci 2011 (CEPCI 2011 603.1)
sofcRefCapacity = 1 # kWe

# Leftover gas burner EC and peak burner EC  ************************ INSTALLED COST *************************

year = [2015, 2020, 2030, 2050]
refCost = [0.06, 0.06, 0.05, 0.05] # milion euro 2015
gasBurnerRefCost = np.interp(evalYear, year, refCost, left=None, right=None, period=None)*1000000 # Euro 2015
gasBurnerRefCepci = 556.8
gasBurnerRefCapacity = 1000 # kW heat generation 
gasBurnerScalingFactor = 1 # Data is valuable up to 10 MW

# SOEC EC = Same as SOFC

# Heat exchanger EC

htxRefCost = 70000 # USD Rules of Thumb
htxRefCepci = 1000 
htxRefCapacity = 100 # m2 transfer area
htxScalingFactor = 0.71
htxRefLM = 2.5 # L+M* factor Rules of Thumb (2.2-2.8) but many installations
htxRefIns = ecScaling(40000, 1, 1000, USD2CAD, cepciCurrent, 1, 1) # 40 000 for heat exch instrumentation

# COMBUSTION SYSTEM Biomass (propane burner considered already installed)

# Biomass ************************ INSTALLED COST *************************

year = [2015, 2020, 2030, 2050]
refCost = [0.71, 0.69, 0.66, 0.59]
biomassBurnerRefCost = np.interp(evalYear, year, refCost, left=None, right=None, period=None)*1000000 # Euro 2015
biomassBurnerRefCepci = 556.8
biomassBurnerRefCapacity = 6800 # kW heat generation 
biomassBurnerScalingFactor = 1 # For the moment size data is valuable up to 20 mW


# Initiation of results df

df_polygenEC = pd.DataFrame()
df_polygenFIC = pd.DataFrame()
df_combustionEC = pd.DataFrame()

for i in range(0, len(polygenNominalPower)):
    
    # Gasifier EC (Mean of two references)

    gasCapacity1 = polygenNominalPower[i]/woodOdmtHhv # gasifier capacity for ref 1 (odmt/h)
    gasCapacity2 = polygenNominalPower[i] # gasifier power number 2
    gasCost1 = ecScaling(gasRef1Cost, gasRef1Capacity, gasRef1Cepci, SEK2CAD, cepciCurrent, gasScalingFactor, gasCapacity1)
    gasCost2 = ecScaling(gasRef2Cost, gasRef2Capacity, gasRef2Cepci, EUR2CAD, cepciCurrent, gasScalingFactor, gasCapacity2)
    df_polygenEC.loc[i, 'Gasifier ref1 odmt/h'] = gasCapacity1
    df_polygenEC.loc[i, 'Gasifier ref2 kW'] = gasCapacity2
    df_polygenEC.loc[i, 'Gasifier EC'] =  gasCost1 # Keeping only GobiGas cost np.mean([gasCost1, gasCost2])

    # Cyclone + cyclone fan EC

    cycloneCapacity = 7.32*polygenNominalPower[i]/wood45Hhv/3600*1000 # 7.32 m3 per kg of moist biomass (45% mc) per sec
    cycloneFanCapacity = max([7.32*polygenNominalPower[i]/wood45Hhv/3600*1000, 2]) # Minimum of 2
    df_polygenEC.loc[i, 'Cyclone V Flow'] = cycloneCapacity
    df_polygenEC.loc[i, 'Cyclone EC'] = ecScaling(cycloneRefCost, cycloneRefCapacity, cycloneRefCepci, USD2CAD, cepciCurrent, cycloneScalingFactor, cycloneCapacity)
    df_polygenEC.loc[i, 'Cyclone fan V Flow'] = cycloneFanCapacity
    df_polygenEC.loc[i, 'Cyclone fan EC'] = ecScaling(cycloneFanRefCost, cycloneFanRefCapacity, cycloneFanRefCepci, USD2CAD, cepciCurrent, cycloneFanScalingFactor, cycloneFanCapacity)

    # Nickel fluidized bed EC

    NiFluidizedBedCapacity = (7.32+0.056*steamVNi)*polygenNominalPower[i]/wood45Hhv/3600*1000*0.36 # 7.32 m3 + steam per kg of moist biomass (45% mc) For 0.36 second of residence time (Carbon Deposition on Nickel-based Catalyst during Bio..
    df_polygenEC.loc[i, 'Ni fluidized bed Volume'] = NiFluidizedBedCapacity
    df_polygenEC.loc[i, 'Ni fluidized bed EC'] = ecScaling(NiFluidizedBedRefCost, NiFluidizedBedRefCapacity, NiFluidizedBedRefCepci, USD2CAD, cepciCurrent, NiFluidizedBedScalingFactor, NiFluidizedBedCapacity)

    # ZnO guard bed EC

    ZnOGuardBedCapacity = (7.32*(tempZnO/tempNi)+0.056*steamVZnO)*polygenNominalPower[i]/wood45Hhv/3600*1000 # 7.32 m3 (Temp adusted V) + steam V per kg of moist biomass (45% mc)
    df_polygenEC.loc[i, 'ZnO guard bed Volume'] = ZnOGuardBedCapacity
    df_polygenEC.loc[i, 'ZnO guard bed EC'] = ecScaling(ZnOGuardBedRefCost, ZnOGuardBedRefCapacity, ZnOGuardBedRefCepci, EUR2CAD, cepciCurrent, ZnOGuardBedScalingFactor, ZnOGuardBedCapacity)

    # Flash separator EC

    flashLiquidVolume = (flashLiquidProp.v*0.94*polygenNominalPower[i]/wood45Hhv/3600*1000*450) # Liquid with res time = 450 % 0.94 kg/s per kg biomass water out per sec
    flashGasFlow = polygenNominalPower[i]/wood45Hhv/3600*1000*1.036/flashGasDensity # m3/s per kg of moist biomass per sec
    flashGasVolume = max(flashGasFlow/0.35, 0.3) # Minimum of 0.3 m3
    # Find diameter for H = 1.5 x D (solving equations)
    def equations(vars):
        D, H = vars
        eq1 = 1.5*D/H-1 # H = 1.5D
        eq2 = math.pi*(D/2)**2/flashGasVolume-1 # Relation with Volume
        return [eq1, eq2]

    D, H =  fsolve(equations, (1, 1))
    flashAddVolume = (D/2)**2*(0.18+0.15) # Addition of the central part of 18 cm
    flashTotalVolume = max(1, flashAddVolume+flashGasVolume+flashLiquidVolume) # Minimum of 1 m3 for pricing
    df_polygenEC.loc[i, 'Knockout pot volume'] = flashTotalVolume
    df_polygenEC.loc[i, 'Knockout pot EC'] = ecScaling(flashRefCost, flashRefCapacity, flashRefCepci, USD2CAD, cepciCurrent, flashScalingFactor, flashTotalVolume)

    # MeOH synthesis subsystem EC

    compressorPower = polygenNominalPower[i]*0.087 # 8.7 percent of biomass input to compressor work
    df_polygenEC.loc[i, 'Compressor Power'] = compressorPower
    df_polygenEC.loc[i, 'Compressor EC'] = ecScaling(compressorRefCost, compressorRefCapacity, compressorRefCepci, USD2CAD, cepciCurrent, compressorScalingFactor, compressorPower)
    meohBedCapacity = (1.0527/flashGasDensity)*polygenNominalPower[i]/wood45Hhv/3600*1000*0.36 # 1.0527 kg converted to volume per kg of moist biomass (45% mc) For GHSV (10000 h-1)
    df_polygenEC.loc[i, 'MeOH reactor volume'] = meohBedCapacity
    df_polygenEC.loc[i, 'MeOH reactor EC'] = ecScaling(meohRefCost, meohRefCapacity, meohRefCepci, USD2CAD, cepciCurrent, meohScalingFactor, meohBedCapacity)

    # SOFC EC

    sofcCapacity = polygenNominalPower[i]*0.339 # 33.9 % of biomass input to electricity
    df_polygenEC.loc[i, 'SOFC power kW'] = sofcCapacity
    df_polygenEC.loc[i, 'SOFC EC'] = sofcRefCost*sofcCapacity/sofcRefCapacity*cepciCurrent/sofcRefCepci*USD2CAD

    # Leftover gas burner EC

    leftoverBurnerCapacity = polygenNominalPower[i]*0.34
    df_polygenEC.loc[i, 'Leftover burner power kW'] = leftoverBurnerCapacity
    df_polygenEC.loc[i, 'Leftover burner EC'] = ecScaling(gasBurnerRefCost, gasBurnerRefCapacity, gasBurnerRefCepci, EUR2CAD, cepciCurrent, gasBurnerScalingFactor, leftoverBurnerCapacity)

    # SOEC EC

    soecCapacity = polygenNominalPower[i]*0.089 # 8.9 % of biomass input as electricity input
    df_polygenEC.loc[i, 'SOEC power kW'] = soecCapacity
    df_polygenEC.loc[i, 'SOEC EC'] = sofcRefCost*soecCapacity/sofcRefCapacity*cepciCurrent/sofcRefCepci*USD2CAD

    # Peak burner  EC
    peakBurnerCapacity = peakMeOHBurnerPower[polygenNominalPower[i]]
    df_polygenEC.loc[i, 'Peak burner power kW'] = peakBurnerCapacity
    df_polygenEC.loc[i, 'Peak burner EC'] = ecScaling(gasBurnerRefCost, gasBurnerRefCapacity, gasBurnerRefCepci, EUR2CAD, cepciCurrent, gasBurnerScalingFactor, peakBurnerCapacity)

    # Heat exchanger EC
    htxArea = polygenNominalPower[i]/wood45Hhv/3600*1000*327 # Calculated on Open Pinch with the heat streams
    df_polygenEC.loc[i, 'Htx Area'] = htxArea
    df_polygenEC.loc[i, 'Htx EC'] = ecScaling(htxRefCost, htxRefCapacity, htxRefCepci, USD2CAD, cepciCurrent, htxScalingFactor, htxArea)

    # Fixed-investment cost (L+M factors, instrumentation)

    df_polygenEC.loc[i, 'Total FIC'] = df_polygenEC.loc[i, 'Gasifier EC'] \
        + (df_polygenEC.loc[i, 'Cyclone EC']*cycloneRefLM + df_polygenEC.loc[i, 'Cyclone fan EC']*cycloneFanRefLM)*txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc \
            + df_polygenEC.loc[i, 'Ni fluidized bed EC']*NiFluidizedBedRefTM \
                + (df_polygenEC.loc[i, 'ZnO guard bed EC']*ZnOGuardBedRefLM + ZnOGuardBedRefIns)*txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc \
                    + (df_polygenEC.loc[i, 'Knockout pot EC']*flashRefLM + flashRefIns) * txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc \
                        + (df_polygenEC.loc[i, 'Compressor EC']*compressorRefLM+compressorRefIns) * txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc \
                            + (df_polygenEC.loc[i, 'MeOH reactor EC']*meohRefLM + meohRefIns) * txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc \
                                + df_polygenEC.loc[i, 'Leftover burner EC'] \
                                    + df_polygenEC.loc[i, 'SOFC EC'] + df_polygenEC.loc[i, 'SOEC EC'] + df_polygenEC.loc[i, 'Peak burner EC'] \
                                        + (df_polygenEC.loc[i, 'Htx EC']*htxRefLM + htxRefIns) * txFrIn * ctrFee * hmOfFiEx * ctngcy * ctngcy
            # AJOUTER LES AUTRES SOUS-SYSTÈMES
    df_polygenFIC.loc[i, 'Gasifier FIC'] = df_polygenEC.loc[i, 'Gasifier EC']
    df_polygenFIC.loc[i, 'Cyclone FIC'] = (df_polygenEC.loc[i, 'Cyclone EC']*cycloneRefLM + df_polygenEC.loc[i, 'Cyclone fan EC']*cycloneFanRefLM)*txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc
    df_polygenFIC.loc[i, 'Ni fluidized bed FIC'] = df_polygenEC.loc[i, 'Ni fluidized bed EC']*NiFluidizedBedRefTM
    df_polygenFIC.loc[i, 'ZnO guard bed FIC'] = (df_polygenEC.loc[i, 'ZnO guard bed EC']*ZnOGuardBedRefLM + ZnOGuardBedRefIns)*txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc
    df_polygenFIC.loc[i, 'Knockout pot FIC'] = (df_polygenEC.loc[i, 'Knockout pot EC']*flashRefLM + flashRefIns) * txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc
    df_polygenFIC.loc[i, 'Compressor FIC'] = (df_polygenEC.loc[i, 'Compressor EC']*compressorRefLM+compressorRefIns) * txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc
    df_polygenFIC.loc[i, 'MeOH FIC'] = (df_polygenEC.loc[i, 'MeOH reactor EC']*meohRefLM + meohRefIns) * txFrIn*hmOfFiEx*ctrFee*ctngcy*ctngncySc
    df_polygenFIC.loc[i, 'Leftover burner FIC'] = df_polygenEC.loc[i, 'Leftover burner EC']
    df_polygenFIC.loc[i, 'SOFC FIC'] = df_polygenEC.loc[i, 'SOFC EC']
    df_polygenFIC.loc[i, 'SOEC FIC'] = df_polygenEC.loc[i, 'SOEC EC']
    df_polygenFIC.loc[i, 'Peak Burner FIC'] = df_polygenEC.loc[i, 'Peak burner EC']
    df_polygenFIC.loc[i, 'Htx FIC'] = (df_polygenEC.loc[i, 'Htx EC']*htxRefLM + htxRefIns) * txFrIn * hmOfFiEx * ctrFee * ctngcy * ctngcy
    df_polygenFIC.loc[i, 'Total FIC'] = sum(df_polygenFIC.iloc[i, 0:11])


for i in range(0, len(combustionNominalPower)):
    # Biomass burner EC
    df_combustionEC.loc[i, 'Biomass burner EC'] = ecScaling(biomassBurnerRefCost, biomassBurnerRefCapacity, biomassBurnerRefCepci, EUR2CAD, \
        cepciCurrent, biomassBurnerScalingFactor, combustionNominalPower[i])
    if peakPowerDemand > combustionNominalPower[i]:
        peakBurnerCapacity = math.ceil((peakPowerDemand-combustionNominalPower[i])*10**-2)*10**2
    else:
        peakBurnerCapacity = 0
    df_combustionEC.loc[i, 'Peak burner EC'] = ecScaling(gasBurnerRefCost, gasBurnerRefCapacity, gasBurnerRefCepci, EUR2CAD, cepciCurrent, gasBurnerScalingFactor, peakBurnerCapacity)  
    df_combustionEC.loc[i, 'Total FIC'] = df_combustionEC.loc[i, 'Biomass burner EC'] + df_combustionEC.loc[i, 'Peak burner EC']

df_polygenEC.to_csv('CSV/polygenEC.csv')
df_combustionEC.to_csv('CSV/combustionEC.csv')

d = 6

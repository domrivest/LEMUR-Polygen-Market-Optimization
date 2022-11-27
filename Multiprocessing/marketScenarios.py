""" Optimisation for the multiple market scenarios
Author : Dominic Rivest
Highest nominal power (8000 kW), varying electricity, methanol and hydrogen prices
Created in 2022 """

import numpy as np
import pandas as pd
import pyomo.environ as pe
import pyomo.opt as po
import multiprocessing

# Market scenarios definition
seriesOpti =[]
pricesGen =[]
priceIndex = {'eL': 0.11, 'eM': 0.17, 'eH': 0.26, 'mL': 0.065, 'mM': 0.1085, 'mH': 0.1518, 'h2L': 0.1202, 'h2M': 0.2401, 'h2H': 0.3430}

for ElecScen in ['eL', 'eM', 'eH']: # Low, median and high electricity market price
    for MeOHScen in ['mL', 'mM', 'mH']: # Low, median and high methanol market price
        for H2Scen in ['h2L', 'h2M', 'h2H']: # Low, median and high hydrogen market price
            seriesOpti.append(ElecScen+MeOHScen+H2Scen)
            pricesGen.append({'Biomass' : 0.0171, 'Elec': priceIndex[ElecScen], 'MeOH': priceIndex[MeOHScen], 'H2': priceIndex[H2Scen], 'Heat': 0})

tau = 0.25 # Time step of simulation (hours)
power = np.multiply([1600, 3200, 4800, 6400, 8000], tau) # The nominal biomass input power of the system kWh/15 minu
nominalPower = power[4] # Highest system power (8000 kW) for market analysis
MeOHkWh2L = 1/(18.2/3.6) # HHV MeOH
 # Conversion ratio of MeOH kWh to MeOH L
BiomasskWh2T = 1/((14/3.6)*10**3) # Conversion ratio of Biomass kWh HHV to Biomass MOIST tons
TRNSYSInfo = pd.read_excel('Serre_plant-condensation-control_building.xlsx')  # Greenhouse heat demand (kJ/h)
heatDemandIni = TRNSYSInfo['MW pour 20 000 m2']*1000*tau # Conversion from MW to kW (mettre *tau pour kWh)
peakDemand = max(heatDemandIni)  # Peak heat demand (kWh)
operatingHours = 8000 # System annual operation hours
MeOHBurnerEta = 0.822 # Efficiency of the meOh burner
duration = len(heatDemandIni)-1 # duration of the simulation (35040) (test à 100)
maintenanceBreakStart =  int(4344/tau) # Start of the maintenance break (July)
maintenanceBreakEnd = maintenanceBreakStart + int((8760-operatingHours)/tau) # End of the maintenance break (August)
peakMeOHBurnerPower = {1600*tau:5000*tau, 3200*tau:4000*tau, 4800*tau:2500*tau, 6400*tau:1500*tau, 8000*tau:1000*tau}

heatDemandIni = heatDemandIni[0 : duration] # Égalisation de la longueur du vecteur avec le temps


# System conversion efficiencies depending on the operating mode (1-max elec, 2 - max MeOH, 3 - max H2, 4 - Max Thermal)

etaSys = {'Elec': {1: 0.161, 2: 0, 3: 0, 4: 0},
            'MeOH': {1: 0.122, 2:0.141, 3: 0.111, 4: 0.122},
            'H2': {1: 0, 2: 0, 3: 0.319, 4:0},
            'Heat': {1: 0.566, 2: 0.708, 3: 0.420, 4: 0.717}}

def polygenOpti(serieName, prices): # Optimisation of polygeneration in different market scenarios
    # DF for results storage

    optimalValues = pd.DataFrame()
    optimalValuesSum = pd.DataFrame() # Sum of optimal Values of every optimization instance
    optimalValuesTotal = pd.DataFrame() # Grand total

    # Initialisation of optimal products
    optimalValues['Biomass'] = 0
    optimalValues['Elec'] = 0
    optimalValues['MeOH'] = 0
    optimalValues['H2'] = 0
    optimalValues['Heat'] = 0

    # Optimization section

    for instance in range(0, int(np.floor(duration/1000)+1)):

        print("Boucle numéro "+str(instance)) # loop number indication
        # Adjustments for the last loop instance

        model = pe.ConcreteModel()

        if instance == int(np.floor(duration/1000)):
            heatDemand = heatDemandIni[instance*1000 : duration] # HeatDemand for the optimization instance
            model.time = pe.Set(initialize=pe.RangeSet(instance*1000,  duration-1)) # Time for the optimization instance (-1 beacause of RangeSet)
        else:
            heatDemand = heatDemandIni[instance*1000 : instance*1000+1000] # HeatDemand for the optimization instance
            model.time = pe.Set(initialize=pe.RangeSet(instance*1000, instance*1000+1000-1)) # Time for the optimization instance (-1 beacause of RangeSet)


        # Definition of sets
        inputs = {'Biomass', 'MeOH'}
        products = {'Elec', 'MeOH', 'H2', 'Heat'}
        operatingMode = {1, 2, 3, 4}

        model.inputs = pe.Set(initialize=inputs)
        model.products = pe.Set(initialize=products)
        model.operatingMode = pe.Set(initialize=operatingMode)

        # Definition of parameters

        inputPrices = {'Biomass': prices['Biomass'], 'MeOH' : prices['MeOH']}  # Input prices ($/kWh)
        productPrices = {'Elec': prices['Elec'], 'MeOH': prices['MeOH'], 'H2': prices['H2'], 'Heat': prices['Heat']}  # Product prices ($/kWh)

        model.heatDemand = pe.Param(model.time, initialize=heatDemand)
        model.productPrices = pe.Param(model.products, initialize=productPrices)
        model.inputPrices = pe.Param(model.inputs, initialize=inputPrices)

        # Definition of variables

        model.x = pe.Var(model.inputs, model.time, domain=pe.Reals, bounds=(0, None))  # Inputs variables (kWh)
        model.y = pe.Var(model.products, model.operatingMode, model.time, domain=pe.Reals, bounds=(0, None))  # Output variables (kWh)
        model.z = pe.Var(model.operatingMode, model.time, domain=pe.Binary)  # Operating modes (1 on, 0 off)

        # Definition of constraints¸

        model.heatSupply = pe.ConstraintList()  # Heat demand must be satisfied
        for t in model.time:
            lhs = sum(model.y['Heat', :, t])+ model.x['MeOH', t]*MeOHBurnerEta # Heat represents the heat from the polygenSystem
            rhs = model.heatDemand[t] # si peakDemand = Utilisation de la puissance max constante
            model.heatSupply.add(lhs >= rhs)

        model.maxBiomassPower = pe.ConstraintList() # Max heating power
        for t in model.time:
            lhs = model.x['Biomass', t]
            rhs = nominalPower # si nominal power = Utilisation de la puissance max constante
            model.maxBiomassPower.add(lhs <= rhs)

        model.maxMeOHPower = pe.ConstraintList() # Max heating power Methanol
        for t in model.time:
            lhs = model.x['MeOH', t]*MeOHBurnerEta
            rhs = peakMeOHBurnerPower[nominalPower]
            model.maxMeOHPower.add(lhs <= rhs)

        model.conversion = pe.ConstraintList() # Link between biomass and conversion
        for t in model.time:
            for pro in model.products:
                for op in model.operatingMode:
                    lhs = model.y[pro, op, t]
                    rhs = model.x['Biomass', t]*etaSys[pro][op]
                    model.conversion.add(lhs<=rhs)

        model.modeActivation = pe.ConstraintList() # Link between products and operating modes
        for t in model.time:
            for op in model.operatingMode:
                lhs = sum(model.y[:, op, t])
                rhs = model.z[op, t]*nominalPower*2
                model.modeActivation.add(lhs <= rhs)

        model.singleOperatingMode = pe.ConstraintList()  # Single operating mode at at time
        for t in model.time:
            lhs = sum(model.z[:, t])
            rhs = 1
            model.singleOperatingMode.add(lhs <= rhs)

        model.maintenanceBreak = pe.ConstraintList() # July mainteance break (respect 8000 hours of annual operation)
        for t in model.time:
            if t >= maintenanceBreakStart and t <= maintenanceBreakEnd:
                lhs = model.x['Biomass', t]
                rhs = 0
                model.maintenanceBreak.add(lhs <= rhs)

        # Objective function

        expr = sum(sum(model.x[i, t]*model.inputPrices[i] for i in model.inputs) - sum(sum(model.y[i, op, t]*model.productPrices[i]
                                                                            for i in model.products) for op in operatingMode) for t in model.time)
        model.objective = pe.Objective(sense=pe.minimize, expr=expr)

        # Solving

        solver = po.SolverFactory('scip')#("scip")
        results = solver.solve(model, tee=True)
        #solver=po.SolverFactory('cbc', executable='/usr/bin/cbc').solve(model, tee=True).write()

        # Storing results

        for t in model.time:
            optimalValues.loc[t, 'Elec'] = sum(pe.value(model.y['Elec', o, t]) for o in operatingMode)
            optimalValues.loc[t, 'MeOH'] = sum(pe.value(model.y['MeOH', o, t]) for o in operatingMode)
            optimalValues.loc[t, 'H2'] = sum(pe.value(model.y['H2', o, t]) for o in operatingMode)
            optimalValues.loc[t, 'Heat'] = sum(pe.value(model.y['Heat', o, t]) for o in operatingMode)
            optimalValues.loc[t, 'Biomass'] = pe.value(model.x['Biomass', t])
            optimalValues.loc[t, 'MeOHInput'] = pe.value(model.x['MeOH', t])
            optimalValues.loc[t, 'operatingMode1'] = pe.value(model.z[1, t])
            optimalValues.loc[t, 'operatingMode2'] = pe.value(model.z[2, t])
            optimalValues.loc[t, 'operatingMode3'] = pe.value(model.z[3, t])
            optimalValues.loc[t, 'operatingMode4'] = pe.value(model.z[4, t])

        

        # Sum

        optimalValuesSum.at[instance, 'Biomass'] = sum(optimalValues.loc[instance*1000:instance*1000+1000,'Biomass']) # kWh
        optimalValuesSum.at[instance, 'BiomassCost'] = optimalValuesSum.at[instance, 'Biomass']*prices['Biomass'] # $
        optimalValuesSum.at[instance, 'BiomassTons'] = optimalValuesSum.at[instance, 'Biomass']*BiomasskWh2T # Tons

        optimalValuesSum.at[instance, 'MeOHInput'] = sum(optimalValues.loc[instance*1000:instance*1000+1000,'MeOHInput']) # kWh
        optimalValuesSum.at[instance, 'MeOHLitersIn'] = optimalValuesSum.at[instance, 'MeOHInput']*MeOHkWh2L # L
        optimalValuesSum.at[instance, 'MeOHCostIn'] = optimalValuesSum.at[instance, 'MeOHInput']*prices['MeOH'] # $

        optimalValuesSum.at[instance, 'Elec'] = sum(optimalValues.loc[instance*1000:instance*1000+1000,'Elec']) # kWh
        optimalValuesSum.at[instance, 'ElecRevenue'] = optimalValuesSum.at[instance, 'Elec']*prices['Elec'] # $

        optimalValuesSum.at[instance, 'MeOH'] = sum(optimalValues.loc[instance*1000:instance*1000+1000,'MeOH']) # kWh
        optimalValuesSum.at[instance, 'MeOHRevenue'] = optimalValuesSum.at[instance, 'MeOH']*prices['MeOH'] # $
        optimalValuesSum.at[instance, 'MeOHLiters'] = optimalValuesSum.at[instance, 'MeOH']*MeOHkWh2L # L

        optimalValuesSum.at[instance, 'H2'] = sum(optimalValues.loc[instance*1000:instance*1000+1000,'H2']) # kWh
        optimalValuesSum.at[instance, 'H2Revenue'] = optimalValuesSum.at[instance, 'H2']*prices['H2'] # $

        optimalValuesSum.at[instance, 'Heat'] = sum(optimalValues.loc[instance*1000:instance*1000+1000,'Heat']) # kWh
        optimalValuesSum.at[instance, 'HeatRevenue'] = optimalValuesSum.at[instance, 'Heat']*prices['Heat'] # $

        optimalValuesSum.at[instance, 'ObjectiveFunction'] = pe.value(model.objective) # $

        # End of the optimization loop



    optimalValuesTotal.at[0, 'Biomass'] = sum(optimalValuesSum['Biomass']) # kWh
    optimalValuesTotal.at[0, 'BiomassCost'] = sum(optimalValuesSum['BiomassCost']) # $
    optimalValuesTotal.at[0, 'BiomassTons'] = sum(optimalValuesSum['BiomassTons']) # Tons

    optimalValuesTotal.at[0, 'MeOHInput'] = sum(optimalValuesSum['MeOHInput']) # kWh
    optimalValuesTotal.at[0, 'MeOHLitersIn'] = sum(optimalValuesSum['MeOHLitersIn']) # L
    optimalValuesTotal.at[0, 'MeOHCostIn'] = sum(optimalValuesSum['MeOHCostIn']) # $

    optimalValuesTotal.at[0, 'Elec'] = sum(optimalValuesSum['Elec']) # kWh
    optimalValuesTotal.at[0, 'ElecRevenue'] = sum(optimalValuesSum['ElecRevenue']) # $

    optimalValuesTotal.at[0, 'MeOH'] = sum(optimalValuesSum['MeOH']) # kWh
    optimalValuesTotal.at[0, 'MeOHRevenue'] = sum(optimalValuesSum['MeOHRevenue']) # $
    optimalValuesTotal.at[0, 'MeOHLiters'] = sum(optimalValuesSum['MeOHLiters']) # L

    optimalValuesTotal.at[0, 'H2'] = sum(optimalValuesSum['H2']) # kWh
    optimalValuesTotal.at[0, 'H2Revenue'] = sum(optimalValuesSum['H2Revenue']) # $

    optimalValuesTotal.at[0, 'Heat'] = sum(optimalValuesSum['Heat']) # kWh
    optimalValuesTotal.at[0, 'HeatRevenue'] = sum(optimalValuesSum['HeatRevenue']) # $

    optimalValuesTotal.at[0, 'ObjectiveFunction'] = sum(optimalValuesSum['ObjectiveFunction']) # $

    # saving the dataframes
    optimalValues.to_csv('CSV/optimalValues' + serieName + '(' + str(int(nominalPower/tau)) + 'kW).csv')
    optimalValuesSum.to_csv('CSV/optimalValuesSum' + serieName + '(' + str(int(nominalPower/tau)) + 'kW).csv')
    optimalValuesTotal.to_csv('CSV/optimalValuesTotal' + serieName + '(' + str(int(nominalPower/tau)) + 'kW).csv')

if __name__ == '__main__':
    pool = multiprocessing.Pool() #use all available cores, otherwise specify the number you want as an argument
    for i in range(len(seriesOpti)):
        pool.apply_async(polygenOpti, args=(seriesOpti[i], pricesGen[i]))
    pool.close()
    pool.join()

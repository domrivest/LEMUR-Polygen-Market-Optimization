""" Main file of the combustion-based system optimisation
Author : Dominic Rivest
Power is defined variable, system maintenance is taken into account (july break), propane peak burner activated, varying system sizes
Created in 2022 """

import numpy as np
import pandas as pd
import pyomo.environ as pe
import pyomo.opt as po

# Inputs

prices = {'Biomass' : 0.0171, 'Heat': 0, 'Propane': 0.1711} # $/kWh 100 $/tonneAnhydre biomasse, 0,17916 $/kWh LHV propane
biomasskWh2T = 1/((12.1/3.6)*10**3) # Conversion ratio of Biomass kWh to Biomass moist tons (14 HHV, 12.1 LHV)
propanekWh2L = 1/(23.1/3.6) # At 25.1 MJ/L HHV (23.1 MJ/L LHV)
tau = 0.25 # Time step of simulation (hours)
TRNSYSInfo = pd.read_excel('Serre_plant-condensation-control_building.xlsx')  # Greenhouse heat demand (kJ/h)
heatDemandIni = TRNSYSInfo['MW pour 20 000 m2']*1000*tau # Conversion from MW to kW (mettre *tau pour kWh)
peakDemandIni = max(heatDemandIni)  # Peak heat demand (kWh)
operatingHours = 8000 # System annual operation hours (test à 15)
propaneBurnerEta = 0.874 # Efficiency of the propane burner (0.95 LHV)
biomassBurnerEta = 0.778 # Efficiency of the biomass burner (0.9 LHV)
biomassMaxDeltaT = 1 # Maximum biomass power variation over a timestep (15 minutes)
duration = len(heatDemandIni)-1 # duration of the simulation (35040) (test à 100)
maintenanceBreakStart =  int(4344/tau) # Start of the maintenance break (July)
maintenanceBreakEnd = maintenanceBreakStart + int((8760-operatingHours)/tau) # End of the maintenance break (August)
biomassNominalPower = np.multiply(range(1300, 1300*5+1, 1300), tau) # kW*h = kW per 15 min
biomassMinPower = 0.2 # 20 % of the nominal power

heatDemand = heatDemandIni[0 : duration] # Égalisation de la longueur du vecteur avec le temps

for i in range(int(4344/tau), int(5104/tau)) : # The system demand is put artificially to zero for the month of July (760 hours) to enable maintenance stop
    heatDemand[i] = 0

# Optimisation de la portion fournie par le système de biomasse et celle fournie par les brûleurs additionnels au propane
instance = 0
optimalValuesTotal = pd.DataFrame() # Grand total result storage

for nominalPower in biomassNominalPower: # For loop of nominal power
    
    # DF for instance results storage
    optimalValues = pd.DataFrame()


    # Initialisation of optimal products
    optimalValues['Biomass'] = []
    optimalValues['Propane'] = []

    # Optimization section

    print("Nominal power "+str(nominalPower)) # biomass nominal power loop indication

    model = pe.ConcreteModel()

    # Definition of sets
    inputs = {'Biomass', 'Propane'}
    products = {'Heat'}

    model.inputs = pe.Set(initialize=inputs)
    model.products = pe.Set(initialize=products)
    model.time = pe.Set(initialize=pe.RangeSet(0,  duration-1))

    # Definition of parameters

    inputPrices = {'Biomass': prices['Biomass'], 'Propane' : prices['Propane']}  # Input prices ($/kWh)
    productPrices = {'Heat': prices['Heat']}  # Product prices ($/kWh)

    model.heatDemand = pe.Param(model.time, initialize=heatDemand)
    model.productPrices = pe.Param(model.products, initialize=productPrices)
    model.inputPrices = pe.Param(model.inputs, initialize=inputPrices)

    # Definition of variables

    model.x = pe.Var(model.inputs, model.time, domain=pe.Reals, bounds=(0, None))  # Inputs variables (kWh)
    model.y = pe.Var(model.products, model.time, domain=pe.Reals, bounds=(0, None))  # Output variables (kWh)

    # Definition of constraints¸

    model.heatSupply = pe.ConstraintList()  # Heat demand must be satisfied
    for t in model.time:
        lhs = model.y['Heat', t] # Heat represents the heat from both burners (biomass and propane)
        rhs = model.heatDemand[t]
        model.heatSupply.add(lhs >= rhs)

    model.maxBiomassPower = pe.ConstraintList() # Biomass Max heating power
    for t in model.time:
        lhs = model.x['Biomass', t]*biomassBurnerEta
        rhs = nominalPower # si nominal power = Utilisation de la puissance max constante
        model.maxBiomassPower.add(lhs <= rhs)

    model.biomassMinPower = pe.ConstraintList() # Min heating power (20 %)
    for t in model.time:
        if heatDemand[t] > 0:
            lhs = model.x['Biomass', t]
            rhs = nominalPower*biomassMinPower 
            model.biomassMinPower.add(lhs >= rhs)

    model.conversion = pe.ConstraintList() # Link between biomass, propane and heat
    for t in model.time:
        lhs = model.y['Heat', t]
        rhs = model.x['Biomass', t]*biomassBurnerEta+model.x['Propane', t]*propaneBurnerEta
        model.conversion.add(lhs<=rhs)

    model.maintenanceBreak = pe.ConstraintList() # July mainteance break (respect 8000 hours of annual operation)
    for t in model.time:
        if t >= maintenanceBreakStart and t <= maintenanceBreakEnd:
            lhs = model.x['Biomass', t]
            rhs = 0
            model.maintenanceBreak.add(lhs <= rhs)

    # Objective function

    expr = sum(sum(model.x[i, t]*model.inputPrices[i] for i in model.inputs) - sum(model.y[i, t]*model.productPrices[i]
                                                                        for i in model.products) for t in model.time)
    model.objective = pe.Objective(sense=pe.minimize, expr=expr)

    # Solving

    solver = po.SolverFactory('scip')#("scip")
    results = solver.solve(model, tee=True)
    #solver=po.SolverFactory('cbc', executable='/usr/bin/cbc').solve(model, tee=True).write()

    # Storing results

    for t in model.time:
        optimalValues.loc[t, 'Heat'] = pe.value(model.y['Heat', t])
        optimalValues.loc[t, 'Biomass'] = pe.value(model.x['Biomass', t])
        optimalValues.loc[t, 'Propane'] = pe.value(model.x['Propane', t])

    optimalValuesTotal.at[instance, 'Biomass'] = sum(optimalValues['Biomass']) # kWh
    optimalValuesTotal.at[instance, 'BiomassCost'] = optimalValuesTotal.at[instance, 'Biomass']*prices['Biomass'] # $
    optimalValuesTotal.at[instance, 'BiomassTons'] = optimalValuesTotal.at[instance, 'Biomass']*biomasskWh2T # Tons

    optimalValuesTotal.at[instance, 'Propane'] = sum(optimalValues['Propane']) # kWh
    optimalValuesTotal.at[instance, 'PropaneLiters'] =  optimalValuesTotal.at[instance, 'Propane']*propanekWh2L # L
    optimalValuesTotal.at[instance, 'PropaneCost'] =  optimalValuesTotal.at[instance, 'Propane']*prices['Propane'] # $

    optimalValuesTotal.at[instance, 'Heat'] = sum(optimalValues['Heat']) # kWh
    optimalValuesTotal.at[instance, 'HeatRevenue'] = optimalValuesTotal.at[instance, 'Heat']*prices['Heat'] # $

    optimalValuesTotal.at[instance, 'ObjectiveFunction'] = pe.value(model.objective) # $

    # saving the dataframes
    optimalValues.to_csv('CSV/optimalValuesCombustion' + str(nominalPower) + '(' + str(int(nominalPower/tau)) + 'kW).csv')
    instance = instance + 1 

optimalValuesTotal.to_csv('CSV/optimalValuesTotalCombustion.csv')


Bonsoir = 4
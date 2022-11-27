# LEMUR-Polygen-Market-Optimization
Optimization software based on pyomo for the polygeneration system developped in the LEMUR lab

This repository contains all the necessary files to optimize the annual operation of the polygeneration system developed by Antar and Robert. The market scenarios folder is dedicaded to market scenario assessment with the 8000 kW polygeneration system while the Baseline Scenario (Combustion and Polygen) serves for the comparison of the polygeneration and combustion-based systems.

In both folders, the Serre_plant-condensation-control_building.xlsx file contains the greenhouse heat demand information.

In the market scenarios folder, the file marketScenarios.py runs the optimisation program and saves the results in CSV files in the CSV sub-folder. Those CSV are readed by resultMarketOpti which analyses and plots them.

In the Baseline Scenario (Combustion and Polygen) folder, the optiPolygen.py is used for the polygeneration system optimization while optiCombustion is used for the combustion-based system optimization. resultsTreatment.py combines the analysis and graph creation for both polygeneration and combustion-based systems. sensiOpti.py also is used for the sensibility analysis and also optimizes the polygeneration system results with some 15% variation applied to market prices, while resultSensiOpti.py plots the results of this sensibility analysis. sankeyPolygen.py creates a Sankey of the polygeneration system in maximum electricity mode. costEstimate.py is used to estimate the fixed capital investment for both polygeneration and combustion-based systems. The information is saved in the CSV subfolder as polygenEC.csv and combustionEC.csv

To run the marketScenarios optimization, follow this procedure:
1. If the costEstimate parameters have changed, run costEstimate.py with the new parameters and copy the output polygenEC.csv in the CSV subfolder of Baseline Scenario (Combustion and Polygen) to the CSV subfolder of Market scenarios.
2. Modify the marketScenarios.py and run it
3. Analyse the results with resultMarketOpti.py (and verify the economic parameters at the beginning of the file)

To run Baseline Scenario (Combustion and Polygen) optimization, follow this procedure:
1. Verify the capital parameters in costEstimate.py and run it
2. Verify parameters and run optiPolygen.py
3. Verify parameters and run optiCombustion.py
4. Verify/change the economic parameters in resultsTreatment.py and run it to get the results
6. Verify parameters and run sensiOpti.py
7. Verify/change the economic parameters in resultSensiOpti.py to get the results

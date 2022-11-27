""" Sankey diagram of the polygeneration system in max electricity mode
Author : Dominic Rivest
Created in 2022 """

import plotly.graph_objects as go
import numpy as np

flowValues = [1.064, 1.064+0.15-0.007-0.051-0.055, 0.828, 0.583, 0.089+0.045, 0.34, 0.134, 0.343, 0.311, 0.15, 0.161, 0.122, 0.164, 0.123, 0.187-0.045, 0.064, 0.563]
flowValues = np.divide(flowValues, np.max(flowValues)) # Normalisation à 1
nodeLabels = ["Biomass", "DFB gasifier", "Gas Cleaning", "MeOH Reactor", "SOFC", "SOEC", "Burner", "Heat", "Steam", "MeOH", "Electricity", "Useful heat"]

fig = go.Figure(go.Sankey(
    arrangement = "snap",
    node = {
        "label": nodeLabels,
        'pad':30,
        "color" : "slategray"},  # 10 Pixels
    link = {
        "source": [0, 1, 2, 3, 4, 4, 5, 2, 6, 8, 8, 3, 4, 3, 4, 7, 7],
        "target": [1, 2, 3, 4, 5, 6, 3, 7, 8, 1, 2, 9, 10, 7, 7, 0, 11],
        "value": flowValues,
        "label" : flowValues,
        "color" : ["bisque", "beige", "beige", "beige", "lavender", "beige", "plum", "lightcoral", "lightcoral", "gainsboro", "gainsboro", "moccasin", "lavender", "lightcoral", "lightcoral", "lightcoral", "lightcoral"]
        }))

fig.update_layout(
    font=dict(
        family="Open Sans",
        size=24,
        color="RebeccaPurple"
    )
)

fig.show()

d=2
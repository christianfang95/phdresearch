import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

#Define function to simulate logisitc regression data
def sim_regression_data(sample_size = 1000, 
                        x0 = 4, 
                        x1 = -0.5, 
                        x2 = 1.5, 
                        x3 = 1):
    x1 = np.random.choice(
                     a = [0, 1, 2],
                     size = sample_size,
                     p = [0.66, 0.25, (1 - 0.66 - 0.25)])
    x2 = np.random.choice(
                          a = [0, 1, 2, 3],
                          size = sample_size,
                          p = [0.4, (0.6 * 0.66), (0.25 * 0.6), (0.6 * (1 - 0.66 - 0.25))])
    d = pd.DataFrame()
    d["x1"] = x1
    d["x2"] = x2
    d["res"] = d["x1"].map({0:1, 1:0, 2:1})  
    d["stepres"] = d["x2"].map({0:0, 1:2, 2:1, 3:2})
    stepres_dummy = pd.get_dummies(d["stepres"])
    d["nostep"] = stepres_dummy.loc[:,0]
    d["resstep"] = stepres_dummy.loc[:,1]
    d["nonresstep"] = stepres_dummy.loc[:,2]
    conditions = [
                  (x1 == 1),
                  (x2 == 2)
    ]
    choices = [1, 1]
    d["parttime"] = np.select(conditions, choices)
    conditions = [
                 (d["res"] == 1) & (d["nostep"] == 1), #resident, no step
                 (d["res"] == 1) & (d["nonresstep"] == 1), #resident, non resident
                 (d["res"] == 0) & (d["resstep"] == 1), #non- resident, resident
                 (d["res"] == 1) & (d["resstep"] == 1) #resident, resident
                 ]
    choices = [0, 1, 1, 2]
    d["combis"] = np.select(conditions, choices)
    dummys = pd.get_dummies(d["combis"])
    d["resnostep"] = dummys.loc[:,0]
    d["resnores"] = dummys.loc[:,1]
    d["resres"] = dummys.loc[:,2]
    #Make complicated selections
    conditions = [
                  (d["res"] == 0) & (d["nostep"] == 1), #non res, no step
                  (d["res"] == 0) & (d["nonresstep"] == 1) #non res, non res
    ]
    choices = [1, 1]
    filter = np.select(conditions, choices)
    #Filter out parents with nonresident kids only
    d = d[filter == 0]
    d["y"] = x0 + x1 * d["res"] + x2 * d["resnores"] + x3 * d["resres"] + np.random.normal(size = len(d))
    return d


#
example_data = sim_regression_data()

model = smf.ols("y ~ C(combis) + C(combis)*parttime", data = d)
res = model.fit()
res.summary()


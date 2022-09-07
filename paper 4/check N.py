from cgitb import reset
from time import altzone
import pandas as pd
import numpy as np

data = pd.read_stata("/Volumes/data/Research/FSW/Research_data/SOC/Anne-Rigt Poortman/Nieuwe Families NL/Zielinski/Cohesion paper/NFN_Main_Refresh_Sample_wave_3_v1.dta", convert_categoricals=False)

#Combinations that need to go
N_total = len(data)
#remove singles
data['nocohabpartner']=data['A3L02b'].map({1:0, 2:1, 3:1})
data = data.dropna(subset = ["A3L02b"])
filter1 = data['nocohabpartner'] == 1
data = data[filter1]
N_repartnered = len(data)

#Remove people with nonresident fc and nonresident step if there is a step
data["step"] = data["A3L06"].map({1:1, 2:0})
data = data.dropna(subset = ["A3L06"])
data['nonresstepchild'] = data['A3P10'].map({1:0, 2:1, 3:0, 4:2})
conditions = [
    (data["step"] == 0) & (data["nonresstepchild"] == 0),
    (data["step"] == 0) & (data["nonresstepchild"] == 1),
    (data["step"] == 1) & (data["nonresstepchild"] == 0),
    (data["step"] == 1) & (data["nonresstepchild"] == 1),
    (data["step"] == 1) & (data["nonresstepchild"] == 2)
]
choicelist = [0, 0, 1, 2, 3]
data["stepchildfilter"] = np.select(conditions, choicelist) # 2 means nonres

    #Create filter for nonresident focal child #1 means nonres
data['nonresbiochild'] = data['A3I08'].map({1:0, 2:1, 3:0, 4:2})
data["nonresbiochild"].value_counts()
    #nonresbiochild: 0 = resident/pt resident child, 1 = nonresident child, 2 = child alone
    #stepchildfilter: 0 = no stepchild, 1 = resident/ pt resident stepchild, 2 = nonresident stepchild, 3 = stepchild lives alone

#remove missing values
#biochild var
data = data.dropna(subset = ["A3I08"])
data = data[data["A3I08"] != 98]
#stepchild var
data = data[data["A3P10"] != 98]
conditions = [
                        (data["step"] == 1) & (data["A3P10"].isna() == True),
                        (data["step"] == 1) & (data["A3P10"].isna() == False),
                        (data["step"] == 0) & (data["A3P10"].isna() == True),
                        (data["step"] == 0) & (data["A3P10"].isna() == False),
]
choice = [1, 0, 0, 0]
data["stepmiss"] = np.select(conditions, choice)
data = data[data["stepmiss"] == 0]

#Filter out both nonres kids

data = data[((data["nonresbiochild"] == 1) & (data["stepchildfilter"] == 2)) == False]

# Filter out both alone

data = data[((data["nonresbiochild"] == 2) & (data["stepchildfilter"] == 3)) == False]

#Filter out fc alone step nonres
data = data[((data["nonresbiochild"] == 2) & (data["stepchildfilter"] == 2)) == False]

#Filter out step nonres fc alone
data = data[((data["nonresbiochild"] == 2) & (data["stepchildfilter"] == 1)) == False]


#
pd.crosstab(data["nonresbiochild"], data["stepchildfilter"])

#crosstab
crosstab = pd.crosstab(data["nonresbiochild"], data["stepchildfilter"])

crosstab.rename(columns={0: 'No stepchild', 
                1: '(part-time) resident stepchild',
                2: 'nonresident stepchild', 
                3: 'stepchild alone'}, 
                index={0.0: '(part-time) resident child',
                1.0: 'nonresident child',
                2.0: 'Child lives alone'},
                inplace = True)

#Check missing data on dependent variable
#Recode 88 to missing
data["A3T01_a"] = data["A3T01_a"].replace(88.0, np.nan)
data.A3T01_a.value_counts()
data["A3T01_b"] = data["A3T01_b"].replace(88.0, np.nan)
data.A3T01_b.value_counts()
data["A3T01_c"] = data["A3T01_c"].replace(88.0, np.nan)
data.A3T01_c.value_counts()
data["A3T01_d"] = data["A3T01_d"].replace(88.0, np.nan)
data.A3T01_d.value_counts()

data["dvmiss"] = ((data["A3T01_a"].isna()) &  (data["A3T01_b"].isna()) & (data["A3T01_c"].isna()) & (data["A3T01_d"].isna()))
pd.crosstab(data["dvmiss"], data["nonresbiochild"])
pd.crosstab(data["dvmiss"], data["nonresstepchild"])


data = data[data["dvmiss"] == 0]   

crosstab = pd.crosstab(data["nonresbiochild"], data["stepchildfilter"], margins = True)

crosstab.rename(columns={0: 'No stepchild', 
                1: '(part-time) resident stepchild',
                2: 'nonresident stepchild', 
                3: 'stepchild alone'}, 
                index={0.0: '(part-time) resident child',
                1.0: 'nonresident child',
                2.0: 'Child lives alone'},
                inplace = True)



#cohesion scale
#recode so that higher values mean more cohesion -> a, b, d recode
data["A3T01_a"] = data["A3T01_a"].map({1:5, 2:4, 3:3, 4:2, 5:1})
data["A3T01_b"] = data["A3T01_b"].map({1:5, 2:4, 3:3, 4:2, 5:1})
data["A3T01_d"] = data["A3T01_d"].map({1:5, 2:4, 3:3, 4:2, 5:1})

data["cohes"] = (data["A3T01_a"] + data["A3T01_b"] + data["A3T01_c"] + data["A3T01_d"]) / 4

#Crosstab with values

crosstab2 = pd.crosstab(data["nonresbiochild"], data["stepchildfilter"], 
                       values = data["cohes"],
                       aggfunc = "mean",
                       margins = True)

crosstab2.rename(columns={0: 'No stepchild', 
                1: '(part-time) resident stepchild',
                2: 'nonresident stepchild', 
                3: 'stepchild alone'}, 
                index={0.0: '(part-time) resident child',
                1.0: 'nonresident child',
                2.0: 'Child lives alone'},
                inplace = True)

crosstab2


# ARP idea for new categorization
conditions = [
              (data["nonresbiochild"] == 0) & (data["stepchildfilter"] == 0), # res no step -> 0
              (data["nonresbiochild"] == 0) & (data["stepchildfilter"] == 2), # res, non res step -> 2
              (data["nonresbiochild"] == 0) & (data["stepchildfilter"] == 1), # res, res step -> 1
              (data["nonresbiochild"] == 1) & (data["stepchildfilter"] == 1)  # nonres, res step -> 2
]

#0= residential & no step, 1 = residential & residential, 2 = one res the other nonres
choices = [0, 2, 1, 2]

data["combis"] = np.select(conditions, choices)


reg = smf.ols(formula = "cohes ~ C(combis) + age_child_w3 + A3A01 + age_respondent_w3", data = data)
res = reg.fit()
res.summary()

 #nonresbiochild: 0 = resident/pt resident child, 1 = nonresident child, 2 = child alone
 # #stepchildfilter: 0 = no stepchild, 1 = resident/ pt resident stepchild, 2 = nonresident stepchild, 3 = stepchild lives alone




reg = smf.ols(formula = "cohes ~ C(nonresbiochild) + C(stepchildfilter) + age_child_w3 + A3A01 + age_respondent_w3", data = data)
res = reg.fit()
res.summary()

params = pd.DataFrame(res.params)

#SD (Y)
sd_y = np.std(data["cohes"])
res.params / sd_y



N_bothnonres = sum((data["nonresbiochild"] == 1) & (data["stepchildfilter"] == 2))
N_nonresalone = sum((data["nonresbiochild"] == 1) & (data["stepchildfilter"] == 3))
N_alonenonres = sum((data["nonresbiochild"] == 2) & (data["stepchildfilter"] == 2))

N_remaining = N_repartnered - N_bothnonres - N_nonresalone - N_alonenonres 

N_miss_fc = N_repartnered - sum(data["A3I08"] == 98) - 
N_miss_step = 

N_fc_alone = sum(data["nonresbiochild"] == 2)
N_step_alone = sum(data["stepchildfilter"] == 3)

columns = [N_total, N_repartnered, N_bothnonres, N_nonresalone, N_alonenonres, N_remaining, N_fc_alone, N_step_alone]
labels = ["N total", "N repartnered", "- N both nonresident", " - N fc nonres, step alone", "- N fc alone, step nonres", "N remaining", "N fc alone", "N step alone"]

table = pd.DataFrame(columns, labels)


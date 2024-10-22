import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import os
from dataproc import generate_path_and_name_csv, clean_data, import_data
from scipy import stats


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def log_func(x, a, b, c):
    return c / (1 + a * np.exp(b * x))


def myfunc(x):
  return slope * x + intercept


def myfunc2(x, slope, intercept):
    return slope * x + intercept


def find_ratio(filename):
    df_ratio = pd.read_csv('IE_metric.csv')
    for i in range(len(list(df_ratio['Filename']))):
        name = df_ratio['Filename'][i]
        if str(name) == str(filename):
            ratio = df_ratio['FEV1/FVC'][i]
            judge = df_ratio['Obstruction'][i]
            age = df_ratio['Age'][i]
            ref = df_ratio['Judge_Metric'][i]
            ratio_ref = df_ratio['FEV1/FVC ref'][i]
            return ratio, judge, age, ref, ratio_ref
    else:
        return None, None, None, None, None


directory = 'new DF/'
filename_list = []
i = 0
b_list = []
ratio_list = []
name_list = []
judge_list = []
age_list = []
ref_list = []
ratio_ref_list = []
for filename in os.listdir(directory):
    if i < 8000:
        if filename.endswith(".csv"):

            path, name = generate_path_and_name_csv(directory, filename)
            filename_list.append(name)
            xdata, ydata, coord_b = clean_data(path)
            ratio, judge, age, ref, ratio_ref = find_ratio(name)
            if ratio is None:
                continue
            else:
                try:
                    popt, pcov = curve_fit(exp_func, xdata, ydata)
                    b_list.append(popt[1])
                except:
                    b_list.append(0)
                ratio_list.append(ratio)
                name_list.append(name)
                judge_list.append(judge)
                age_list.append(age)
                ref_list.append(ref)
                ratio_ref_list.append(ratio_ref)
    i += 1

for i in range(len(ratio_list)):
    if np.isnan(ratio_list[i]):
        print(name_list[i])

df_fitted_metrics = pd.DataFrame(
    {'Name': name_list,
     'FEV1/FVC': ratio_list,
     'FEV1/FVC ref': ratio_ref_list,
     'Obstruction': judge_list,
     'Judge Metric': ref_list,
     'Fit_b': b_list,
     'Age': age_list
     }
)
df_fitted_metrics.to_csv('fitted_metric.csv')

ratio_obs = []
ratio_no_obs = []
ratio_unknown = []
age_no_obs = []
age_obs = []
age_unknown = []
b_obs = []
b_no_obs = []
b_unknown = []
name_obs = []
for i in range(len(name_list)):
    if ref_list[i]:
        ratio_obs.append(ratio_list[i])
        b_obs.append(b_list[i])
        name_obs.append(name_list[i])
        age_obs.append(age_list[i])
    else:
        ratio_no_obs.append(ratio_list[i])
        b_no_obs.append(b_list[i])
        age_no_obs.append(age_list[i])


ratio_no_obs_40 = []
ratio_obs_40 = []
b_obs_40 = []
b_no_obs_40 = []
ratio_obs_65 = []
ratio_no_obs_65 = []
b_obs_65 = []
b_no_obs_65 = []
ratio_no_obs_85 = []
ratio_obs_85 = []
b_obs_85 = []
b_no_obs_85 = []
for i in range(len(age_obs)):
    if age_list[i]<=40:
        ratio_obs_40.append(ratio_obs[i])
        b_obs_40.append(b_obs[i])
    elif age_list[i]<=65:
        ratio_obs_65.append(ratio_obs[i])
        b_obs_65.append(b_obs[i])
    else:
        ratio_obs_85.append(ratio_obs[i])
        b_obs_85.append(b_obs[i])
for i in range(len(age_no_obs)):
    if age_list[i]<=40:
        ratio_no_obs_40.append(ratio_no_obs[i])
        b_no_obs_40.append(b_no_obs[i])
    elif age_list[i]<=65:
        ratio_no_obs_65.append(ratio_no_obs[i])
        b_no_obs_65.append(b_no_obs[i])
    else:
        ratio_no_obs_85.append(ratio_no_obs[i])
        b_no_obs_85.append(b_no_obs[i])

slope_40, intercept_40, r_40, p_40, std_err_40 = stats.linregress(b_no_obs_40, ratio_no_obs_40)
# print(p_40)
slope_65, intercept_65, r_65, p_65, std_err_65 = stats.linregress(b_no_obs_65, ratio_no_obs_65)
# print(p_65)
slope_85, intercept_85, r_85, p_85, std_err_85 = stats.linregress(b_no_obs_85, ratio_no_obs_85)
# print(p_85)
slope_all, intercept_all, r_all, p_all, std_err_all = stats.linregress(b_no_obs, ratio_no_obs)
print(p_all)
x_array = np.linspace(0, 6.5)
regression_40 = myfunc2(x_array, slope_40, intercept_40)
regression_65 = myfunc2(x_array, slope_65, intercept_65)
regression_85 = myfunc2(x_array, slope_85, intercept_85)
regression_all = myfunc2(x_array, slope_all, intercept_all)


plt.figure()
plt.scatter(b_obs, ratio_obs, color='C1', alpha=0.33, label='obs')
plt.scatter(b_no_obs, ratio_no_obs, color='C0', alpha=0.33, label='no_obs')
plt.plot(x_array, np.array(regression_40)-1.64*std_err_40*np.sqrt(len(x_array)), c='C2')
plt.plot(x_array, np.array(regression_65)-1.64*std_err_65*np.sqrt(len(x_array)), c='C3')
plt.plot(x_array, np.array(regression_85)-1.64*std_err_85*np.sqrt(len(x_array)), c='C4')
plt.plot(x_array, np.array(regression_all)-1.64*std_err_all*np.sqrt(len(x_array)), c='k')

plt.legend(['Obstruction', 'No Obstruction', '<40', '40-65', '65+', 'All'])
plt.xlabel('Fitted Metric')
plt.ylabel('FEV1/FVC')
# plt.savefig('fitted_metric_age.svg')
plt.show()
# print(np.mean(age_list))
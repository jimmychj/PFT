import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import os


def func(x, a, tau):
    return a * np.exp(-x/tau)


def generate_path_and_name_pdf(directory, filename):
    path = os.path.join(directory, filename)
    string2 = directory
    string3 = '.Pdf'
    name = path.replace(string2, '')
    name = name.replace(string3, '')
    return path, name


def generate_path_and_name_csv(directory, filename):
    path = os.path.join(directory, filename)
    string2 = directory
    string3 = '.csv'
    name = path.replace(string2, '')
    name = name.replace(string3, '')
    return path, name


def func_linear(x, a, b):
    return a * x + b


def import_data(directory):
    dataframe = pd.read_csv(directory)
    dataframe.columns = ['x', 'y']
    return dataframe


def clean_data(file_directory):
    d = import_data(file_directory)
    xdata = d['x']
    ydata = d['y']
    pkind = np.argmax(ydata)
    xpk = xdata[pkind]
    ypk = ydata[pkind]
    x_max = max(xdata)
    x_new = np.linspace(xpk, x_max, 100)
    f = interp1d(xdata[pkind:], ydata[pkind:], kind='linear')
    point_b_x = x_max * 0.75
    point_b_y = f(point_b_x)
    coord_b = [point_b_x, point_b_y]
    y_new = f(x_new)
    # plt.plot(xdata, ydata, 'o', x_new, f(x_new), '-')
    # plt.legend(['data', 'fit'], loc='best')
    # plt.show()
    return x_new, y_new, coord_b


def fit_ab_bc(x_new, y_new, coord_b):
    x_ab = []
    x_bc = []
    for i in x_new:
        if i < coord_b[0]:
            x_ab.append(i)
        else:
            x_bc.append(i)
    y_ab = y_new[:len(x_ab)]
    y_bc = y_new[len(x_ab):]
    x_ab = np.array(x_ab)
    x_bc = np.array(x_bc)
    y_ab = np.array(y_ab)
    y_bc = np.array(y_bc)
    popt_ab, pcov_ab = curve_fit(func_linear, x_ab, y_ab)
    yhat_ab = func_linear(x_ab, popt_ab[0], popt_ab[1])
    # a_ab = popt_ab[0]
    # b_ab = popt_ab[1]
    popt_bc, pcov_bc = curve_fit(func_linear, x_bc, y_bc)
    yhat_bc = func_linear(x_bc, popt_bc[0], popt_bc[1])
    # a_bc = popt_bc[0]
    # b_bc = popt_bc[1]
    # ratio = popt_ab[0] / popt_bc[0]
    degree1 = np.arctan(popt_ab[0])/np.pi * 180
    degree2 = np.arctan(popt_bc[0])/np.pi * 180
    degree = 180 + degree1 - degree2
    # print(degree)
    plt.plot(x_new, y_new)
    plt.scatter(coord_b[0], coord_b[1])
    plt.plot(x_ab, yhat_ab, color='#FF7F50', linewidth=3, alpha=0.7)
    plt.plot(x_bc, yhat_bc, color='#FF7F50', linewidth=3, alpha=0.7)
    plt.xlabel('Volume (L)')
    plt.ylabel('Flow (L/s)')
    # plt.show()
    return degree


def find_csn(filename):
    df_csn = pd.read_csv('name_csn.csv', usecols=['Filename', 'CSN'])
    for i, name in enumerate(df_csn['Filename']):
        if str(name) == filename:
            csn = df_csn['CSN'][i]
            return csn


def create_table(directory):
    filename_list = []
    ratio_list = []
    csn_list = []
    degree_list = []
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".csv"):
            path, name = generate_path_and_name_csv(directory, filename)
            filename_list.append(name)
            csn_list.append(find_csn(name))
            x_new, y_new, coord_b = clean_data(path)
            degree = fit_ab_bc(x_new, y_new, coord_b)
            degree_list.append(degree)
    data = {'Filename': filename_list, 'CSN': csn_list, 'AB/BC': ratio_list, 'Angle': degree_list}
    df_new = pd.DataFrame(data, columns=['Filename', 'CSN', 'AB/BC', 'Angle'])
    return df_new


def main():
    df = create_table('P:/Pro00105365 - Development of an AI-PFT Interpretation Algorithm'
                      '/2020-10-22_Data-Delivery/Digitized_data/')
    df.to_csv('Angle.csv', index=False)
    print(df.head())


if __name__ == '__main__':
    main()


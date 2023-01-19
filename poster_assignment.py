import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.optimize as opt
import errors as err


def data_read():
    ghg_data_read = pd.read_csv("ghg_data.csv", skiprows=4)
    electric_data = pd.read_csv("electric_data.csv", skiprows=4)
    return ghg_data_read, electric_data;

def processing_values(df_final):
    df_final = df_final[['Country Name','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005',
                         '2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']]
    print(df_final)
    years = [1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,
             2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
    y_values = df_final.iloc[:,1:]
    print(y_values)
    y_values.reset_index(drop=True, inplace=True)
    new_df = y_values.transpose()
    print(new_df)
    new_data = pd.DataFrame({'years': years,
                           'values': new_df[0]})
    new_data.reset_index(drop=True, inplace=True)
    print(new_data)
    return new_data;

def preprocessing(df):
    scaler = MinMaxScaler()
    scaler.fit(df[['years']])
    df['years'] = scaler.transform(df[['years']])
    scaler.fit(df[['values']])
    df['values'] = scaler.transform(df[['values']])
    return df;

def find_cluster_plot(new_data, name):
    print(name)
    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(new_data[['years', 'values']])
    new_data['cluster']=y_predicted
    df1 = new_data[new_data.cluster==0]
    print(df1)
    df2 = new_data[new_data.cluster==1]
    df3 = new_data[new_data.cluster==2]
    plt.figure()
    plt.scatter(df1['years'],df1['values'],color='b', label='cluster 1')
    plt.scatter(df2.years,df2['values'],color='red', label='cluster 2')
    plt.scatter(df3.years,df3['values'],color='m', label='cluster 3')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*',label='centroid')
    plt.title('Greenhouse Gas Emission of '+ name + ' (1995-2019)')
    plt.xlabel('Years')
    plt.ylabel('Greenhouse Gas Emission (GHG)')
    plt.legend()
    plt.show()
    return
    
    

df, electric_df = data_read();
df_final = df.loc[(df['Country Name'] == 'Australia')]
new_data_aus = processing_values(df_final)
data_aus = preprocessing(new_data_aus)
find_cluster_plot(data_aus, 'Australia')
df_final = df.loc[(df['Country Name'] == 'Canada')]
new_data_canada = processing_values(df_final)
data_canada = preprocessing(new_data_canada)
find_cluster_plot(data_canada, 'Canada')

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    
    t = t - 1995.0
    f = n0 * np.exp(g*t)
    
    return f


electric_df = electric_df.loc[(electric_df['Country Name'] == 'Indonesia')]
data_electric = processing_values(electric_df)
plt.figure()
#plt.scatter(data_electric['years'],data_electric['values'])
print(type(data_electric["years"]))
print(data_electric)
data_electric["years"] = pd.to_numeric(data_electric["years"])
print(type(data_electric["years"]))
param, covar = opt.curve_fit(exponential, data_electric["years"], data_electric["values"], 
                             p0=(262.016541603514, 0.02))

data_electric["fit"] = exponential(data_electric["years"], *param)
data_electric.plot("years", ["values", "fit"])
plt.show()


sigma = np.sqrt(np.diag(covar))
year = np.arange(1995, 2031)
print(year)
forecast = exponential(year, *param)
low, up = err.err_ranges(year, exponential, param, sigma)

plt.figure()
plt.plot(data_electric["years"], data_electric["values"], label="Electric Power Consumption")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()
















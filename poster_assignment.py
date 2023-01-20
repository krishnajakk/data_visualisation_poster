import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.optimize as opt
import errors as err


def data_read():
    """
    Reading data required for the plots
    read & return greenhouse gas data and its transpose
    read & return electric power consumption data and its transpose
    """
    
    # Reading data using pandas dataframe
    ghg_data_read = pd.read_csv("ghg_data.csv", skiprows=4)
    # Taking the transpose of data
    ghg_data_read_transposed = ghg_data_read.transpose()
    electric_data = pd.read_csv("electric_data.csv", skiprows=4)
    electric_data_transposed = electric_data.transpose()
    return ghg_data_read,ghg_data_read_transposed, electric_data, electric_data_transposed;

def processing_values(df_final):
    """
    Sorting data values for final plot
    sorting the data in a required manner to draw clusters
    """
    
    # Extracting the required columns
    df_final = df_final[['Country Name','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005',
                         '2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']]
    years = [1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,
             2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
    # Required positions using iloc function
    y_values = df_final.iloc[:,1:]
    y_values.reset_index(drop=True, inplace=True)
    new_df = y_values.transpose()
    # creating a new dataframe with the extracted values
    new_data = pd.DataFrame({'years': years,
                           'values': new_df[0]})
    new_data.reset_index(drop=True, inplace=True)
    return new_data;

def preprocessing(df):
    """
    Normalising the data (data transformation)
    using minmaxscaler 
    """
    scaler = MinMaxScaler()
    scaler.fit(df[['years']])
    # Normalising the year column
    df['years'] = scaler.transform(df[['years']])
    scaler.fit(df[['values']])
    # Normalising the values column
    df['values'] = scaler.transform(df[['values']])
    return df;

def find_cluster_plot(new_data, name):
    """
    Computing KMeans and plotting
    find k-means using Kmeans function 
    plotting the graph with data values and cluster centroids
    """
    
    # KMeans function returns the key values for clustering
    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(new_data[['years', 'values']])
    new_data['cluster']=y_predicted
    # Extracting the values with different clusters
    df1 = new_data[new_data.cluster==0]
    df2 = new_data[new_data.cluster==1]
    df3 = new_data[new_data.cluster==2]
    # Plotting 
    plt.figure()
    plt.scatter(df1['years'],df1['values'],color='b', label='cluster 1')
    plt.scatter(df2.years,df2['values'],color='red', label='cluster 2')
    plt.scatter(df3.years,df3['values'],color='m', label='cluster 3')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*',label='centroid')
    # General matters
    plt.title('Greenhouse Gas Emission of '+ name + ' (1995-2019)')
    plt.xlabel('Years')
    plt.ylabel('Greenhouse Gas Emission (GHG)')
    plt.legend()
    plt.show()
    return

def find_sse():
    """
    computing SSE using KMeans function
    """
    kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42,
        }
       
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data_aus)
        sse.append(kmeans.inertia_)
    # Calling the plot_SSE() function in the return itself
    return plot_sse(sse);

def plot_sse(sse):
    """ 
    Plotting SSE versus number of clusters graph
    To find appropriate number of clusters for data collected
    """
    plt.plot(range(1, 11), sse,label="SSE")
    # General matters
    plt.xticks(range(1, 11))
    plt.title("Elbow Method For Australia")
    plt.legend()
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    return

def exponential(year, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    
    year = year - 1995.0
    f = n0 * np.exp(g*year)
    
    return f

# Reading the returned values from data_read() method
df, df_transpose, electric_df, electric_df_transpose = data_read();
# Taking data of Australia 
df_final = df.loc[(df['Country Name'] == 'Australia')]
new_data_aus = processing_values(df_final)
data_aus = preprocessing(new_data_aus)
find_cluster_plot(data_aus, 'Australia')
# Taking data of Canada
df_final = df.loc[(df['Country Name'] == 'Canada')]
new_data_canada = processing_values(df_final)
data_canada = preprocessing(new_data_canada)
find_cluster_plot(data_canada, 'Canada')

# Calling sse method to calculate sse and return the value
sse_values = find_sse();

# Taking Country as Indonesia to plot fitting curve
electric_df = electric_df.loc[(electric_df['Country Name'] == 'Indonesia')]
data_electric = processing_values(electric_df)
data_electric["years"] = pd.to_numeric(data_electric["years"])

# Fitting the curve with the data collected using curve_fit() function
param, covar = opt.curve_fit(exponential, data_electric["years"], data_electric["values"], 
                             p0=(262.016541603514, 0.02))
# Adding column named fit to the dataframe
data_electric["fit"] = exponential(data_electric["years"], *param)
plt.figure()
plt.plot(data_electric["years"], data_electric["values"], label="Electric Power Consumption")
plt.plot(data_electric["years"],data_electric["fit"], label = "fit")
# General matters
plt.title("Electric Power Consumption of Indonesia (1995-2019)")
plt.xlabel("Year")
plt.legend()
plt.ylabel("Electric Power Consumption (kWh per capita)")
plt.show()

# Finding sigma and forecast values for the data
sigma = np.sqrt(np.diag(covar))
year = np.arange(1995, 2031)
forecast = exponential(year, *param)
# Finding error ranges for the data
low, up = err.err_ranges(year, exponential, param, sigma)
plt.figure()
plt.plot(data_electric["years"], data_electric["values"], label="Electric Power Consumption")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
# General matters
plt.title("Forecast of Electric Power Consumption of Indonesia (1995-2030)")
plt.xlabel("year")
plt.ylabel("Electric Power Consumption (kWh per capita)")
plt.legend()
plt.show()
















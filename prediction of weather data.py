import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection 
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
#This function replaces the Snowfall values.  There are null values in the original
#and the value '#VALUE!' in some.
def impute_snowfall(col):
  snow=str(col[0]).replace(" ", "")
  if pd.isnull(snow):
      return float(0)
  elif (snow == '#VALUE!') or (snow == 'nan'):
      return float(0)
  else:
      return float(snow)

#This replaces the null values in the Poor Weather column.  Since they are only 1 or 0
#values I had to split the function out from the generic one below.
def impute_poorweather(col):
  snow=str(col[0]).replace(" ", "")
  if pd.isnull(snow):
      return float(0)
  elif (float(snow) != 0) and (float(snow) != 1):
      return float(0)
  else:
      return float(snow)

#This function replaces any generic columns that only need nulls replaced.      
def impute_misc(col):
  snow=str(col[0]).replace(" ", "")
  if pd.isnull(snow):
      return float(snow).mean()
  elif snow == 'T' or (snow == 'nan'):
      return float(0)
  else:
      return float(snow)
      
              
weather=pd.read_csv("Summary of Weather.csv", dtype={"WindGustSpd": np.str, "Snowfall": np.str, "PoorWeather": np.str, "TSHDSBRSGF": np.str, "SNF": np.str,})

#Applying any of the above functions to required columns.
weather['Precip'] = weather[['Precip']].apply(impute_misc,axis=1)
weather['WindGustSpd'] = weather[['WindGustSpd']].apply(impute_misc,axis=1)
weather['Snowfall'] = weather[['Snowfall']].apply(impute_snowfall,axis=1)
weather['PoorWeather'] = weather[['PoorWeather']].apply(impute_poorweather,axis=1)
weather['PRCP'] = weather[['PRCP']].apply(impute_misc,axis=1)
weather['DR'] = weather[['DR']].apply(impute_misc,axis=1)
weather['SPD'] = weather[['SPD']].apply(impute_misc,axis=1)
weather['MAX'] = weather[['MAX']].apply(impute_misc,axis=1)
weather['MIN'] = weather[['MIN']].apply(impute_misc,axis=1)
weather['MEA'] = weather[['MEA']].apply(impute_misc,axis=1)
weather['SNF'] = weather[['SNF']].apply(impute_misc,axis=1)
weather['SND'] = weather[['SND']].apply(impute_misc,axis=1)
weather['PGT'] = weather[['PGT']].apply(impute_misc,axis=1)
weather['TSHDSBRSGF'] = weather[['TSHDSBRSGF']].apply(impute_poorweather,axis=1)

#These columns are ALL filled with only null values, so I've removed them.
weather.drop(['FT'],axis=1,inplace=True)
weather.drop(['FB'],axis=1,inplace=True)
weather.drop(['FTI'],axis=1,inplace=True)
weather.drop(['ITH'],axis=1,inplace=True)
weather.drop(['SD3'],axis=1,inplace=True)
weather.drop(['RHX'],axis=1,inplace=True)
weather.drop(['RHN'],axis=1,inplace=True)
weather.drop(['RVG'],axis=1,inplace=True)
weather.drop(['WTE'],axis=1,inplace=True)

#This will just output a heatmap showing that all the null values are gone.
#sns.heatmap(weather.isna(),yticklabels=False,cbar=False,cmap='cubehelix')
#plt.show()

x=weather[['Precip','WindGustSpd','Snowfall','PoorWeather','PRCP','DR','SPD','MIN','MEA','SNF','SND','PGT','TSHDSBRSGF']]
y=weather['MAX']

v=0.30# validation size .
s=5 # seed
scoring='accuracy'
xtrain,xtest,ytrain,ytest= model_selection.train_test_split(x,y,test_size=v, random_state=s)




regr = linear_model.LinearRegression()
regr.fit(xtrain,ytrain)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

predictions = regr.predict(xtest)

print(predictions)

z=r2_score(ytest,predictions)
print(z)

plt.scatter(x,y)
plt.plot(xtest, predictions, color='red')
plt.show()




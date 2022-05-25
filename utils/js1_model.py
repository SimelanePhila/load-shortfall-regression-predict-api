# Dependencies
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('./data/df_train.csv')

#Filling missing values
train['Valencia_pressure'].fillna(train['Valencia_pressure'].mean(), inplace = True)

train = train.drop(columns=['time'])

## Splitting our data into dependent Variable and Independent Variable
X = train.drop(columns = 'load_shortfall_3h')
y = train['load_shortfall_3h'].astype('int')


X = X[['Unnamed: 0', 'Madrid_wind_speed', 'Bilbao_rain_1h',
            'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
            'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
            'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
            'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
            'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_pressure',
            'Seville_rain_3h', 'Madrid_rain_1h', 'Barcelona_rain_3h',
            'Valencia_snow_3h', 'Madrid_weather_id', 'Barcelona_weather_id',
            'Bilbao_pressure', 'Seville_weather_id', 'Seville_temp_max',
            'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp',
            'Bilbao_weather_id', 'Seville_temp', 'Valencia_humidity',
            'Valencia_temp_min', 'Barcelona_temp_max', 'Madrid_temp_max',
            'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp',
            'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min',
            'Madrid_temp', 'Madrid_temp_min']]

# Create standardization object
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)


#Separating our models into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state = 7)

model = RandomForestRegressor()
# Fitting the model
model.fit(X_train,y_train)


# Pickle model for use within our API
save_path = '../assets/trained-models/random_forest.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(model, open(save_path,'wb'))
    
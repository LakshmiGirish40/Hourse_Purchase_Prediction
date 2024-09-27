import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import pickle


#Importing DataSet 
House_data = pd.read_csv(r"D:\Data_Science&AI\ClassRoomMaterial\September\6th- Simple_Linear_regression\6th- slr\SLR - House price prediction\House_data.csv")

hd = House_data.copy()

space=hd['sqft_living15']
price=hd['price']

X = np.array(space).reshape(-1, 1)
y = np.array(price)

#h =  hd.drop(columns=['id','date'],axis=1,inplace=True)

House_data.drop(columns=['id','date'],axis=1,inplace=True)
House_data.columns


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Fitting simple linear regression to the Training Set

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(X_train,y_train)

#Predicting the prices

y_pred = regression.predict(X_test)
y_pred


#Visualizing the Test Results 
plt.scatter(X_train,y_train,color='g')

plt.plot(X_train,regression.predict(X_train),'bo',linestyle='dashed')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#correlation
correlation = House_data.corr()
correlation

#understanding the distribution with seaborn
with  sns.plotting_context("notebook",font_scale=2.5):  
    g = sns.pairplot(House_data[['sqft_lot','sqft_above','price','sqft_living15','bedrooms']], hue='bedrooms', palette='tab20',height=6)
g.set(xticklabels=[]);

# Predict salary for 12 and 20 years of experience using the trained model
y_sqrt1 = regression.predict([[3000]])
y_sqrt2 = regression.predict([[2500]])
y_sqrt3 = regression.predict([[1500]])
y_sqrt4 = regression.predict([[1100]])
print(f"Predicted Price for y_sqrt1  of experience: ${y_sqrt1 [0]:,.2f}")
print(f"PredictedPrice for y_sqrt2 of experience: ${y_sqrt2 [0]:,.2f}")
print(f"Predicted Price for y_sqrt3  of experience: ${y_sqrt3 [0]:,.2f}")
print(f"PredictedPrice for y_sqrt4 of experience: ${y_sqrt4 [0]:,.2f}")

#separating independent and dependent variable
X = House_data.iloc[:,1:].values
y = House_data.iloc[:,0].values
# Check model performance
bias = regression.score(X_train, y_train)
variance = regression.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regression.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
filename = 'multi_linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regression, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

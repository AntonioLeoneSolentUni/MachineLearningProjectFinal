import pandas as pd


dataToClean = pd.read_csv('personal_finance_employees_V1.csv')


x = dataToClean["Water Bill (£)"].mean()
dataToClean["Water Bill (£)"].fillna(x, inplace = True)
x = dataToClean["Sky Sports (£)"].mean()
dataToClean["Sky Sports (£)"].fillna(x, inplace = True)
x = dataToClean["Other Expenses (£)"].mean()
dataToClean["Other Expenses (£)"].fillna(x, inplace = True)
x = dataToClean["Monthly Outing (£)"].mean()
dataToClean["Monthly Outing (£)"].fillna(x, inplace = True)
dataToClean["Savings for Property (£)"].fillna(0, inplace = True)




dataToClean.to_csv('CleanedData.csv')
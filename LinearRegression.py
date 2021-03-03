import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('input.csv')
df = data[['rawat_inap', 'isolasi_mandiri']]
df.columns = ['x', 'y']

x_train = df['x'].values[:,np.newaxis]
y_train = df['y'].values

lm = LinearRegression()
lm.fit(x_train,y_train)
print("=========================================================================================")
print(data)
print("=========================================================================================")

pb = lm.predict(x_train)
dfc = pd.DataFrame({'x': df['x'],'y':pb})
plt.scatter(df['x'],df['y'],color='#3467eb')
plt.plot(dfc['x'],dfc['y'],color='#3467eb',linewidth=1)
plt.axis([48, 70, 30, 60])
plt.title('Kasus Covid-19 DKI Jakarta per-Januari 2021\n menggunakan Algoritma Regresi Linier by Willy Santoso')
plt.xlabel('Pasien rawat inap')
plt.ylabel('Pasien isolasi mandiri')
plt.show()
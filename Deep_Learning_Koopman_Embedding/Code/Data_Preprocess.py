import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

D = pd.read_csv("MSD_Obs_Data.csv") 
x1 = D["X1"].tolist()
x2 = D["X2"].tolist() 
x1.pop()
x2.pop()
print("The length is",len(x1),len(x2))

X_info  = np.transpose([x1[:len(x1)-2],x2[:len(x2)-2]])
Xd_info = np.transpose([x1[1:len(x1)-1],x2[1:len(x2)-1]])
print(X_info.shape)
print(Xd_info.shape)
##########################
np.save("X.npy",X_info.tolist())
np.save("Y.npy",Xd_info.tolist())

x_mean = np.mean(X_info)
x_std  = np.std(X_info) 
xd_mean = np.mean(Xd_info)
xd_std = np.std(Xd_info)

Xn_info = (X_info - x_mean)/x_std
Xnd_info = (Xd_info - xd_mean)/xd_std
np.save("Xn.npy",Xn_info.tolist())
np.save("Yn.npy",Xnd_info.tolist())
##########################
plt.plot(Xn_info)
plt.plot(Xnd_info)
plt.show() 




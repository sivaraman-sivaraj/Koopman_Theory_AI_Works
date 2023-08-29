import os
import torch 
import numpy as np
import torch.nn as nn 
import matplotlib.pyplot as plt 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.ipL = nn.Linear(2,128)   # has pre trained weight
        self.HL1 = nn.Linear(128,128) # has pre trained weight
        self.HL2 = nn.Linear(128,128) # has pre trained weight
        self.opL = nn.Linear(128,6)   # output layer
        
    def forward(self, x):
        x = torch.tanh(self.ipL(x))  # sigmiodal,relu,.etc.,
        x = torch.tanh(self.HL1(x))
        x = torch.tanh(self.HL2(x))
        x = torch.tanh(self.opL(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.ipL = nn.Linear(6,128)   # has pre trained weight
        self.HL1 = nn.Linear(128,128) # has pre trained weight
        self.HL2 = nn.Linear(128,128) # has pre trained weight
        self.opL = nn.Linear(128,2)   # output layer
        
    def forward(self, x):
        x = torch.tanh(self.ipL(x))   # sigmiodal,relu,.etc.,
        x = torch.tanh(self.HL1(x))
        x = torch.tanh(self.HL2(x))
        x = self.opL(x)
        return x

class Koopman_Eigen_Matrix(nn.Module):
    def __init__(self):
        super(Koopman_Eigen_Matrix, self).__init__()
        self.ipL = nn.Linear(6,128)   # has pre trained weight
        self.HL1 = nn.Linear(128,128) # has pre trained weight
        self.HL2 = nn.Linear(128,128) # has pre trained weight
        self.opL = nn.Linear(128,6)   # output layer
        
    def forward(self, x):
        x = torch.tanh(self.ipL(x))   # sigmiodal,relu,.etc.,
        x = torch.tanh(self.HL1(x))
        x = torch.tanh(self.HL2(x))
        x = torch.tanh((self.opL(x)))
        return x
###############################################
############# sanity check ####################
###############################################
S_00 = torch.load("Encoder.pt")
phi_x     = Encoder()
with torch.no_grad():
    phi_x .ipL.weight.copy_(S_00['ipL.weight'])
    phi_x .ipL.bias.copy_(S_00['ipL.bias'])
    phi_x .HL1.weight.copy_(S_00['HL1.weight'])
    phi_x .HL1.bias.copy_(S_00['HL1.bias'])
    phi_x .HL2.weight.copy_(S_00['HL2.weight'])
    phi_x .HL2.bias.copy_(S_00['HL2.bias'])
    phi_x .opL.weight.copy_(S_00['opL.weight'])
    phi_x .opL.bias.copy_(S_00['opL.bias'])
phi_x .eval() 

phi_d_x   = Decoder() 
phi_d_x.load_state_dict(torch.load("Decoder.pt"))
phi_d_x.eval()

K_net     = Koopman_Eigen_Matrix()
K_net.load_state_dict(torch.load("Koopman_EV.pt"))
K_net.eval() 
##################################################
##################################################
##################################################
X_actual_ = np.load("Xn.npy")
Y_actual  = np.load("Yn.npy")
X_actual = X_actual_.tolist()[1:2499]

X_pred = list()
print(X_actual[0])
for i in range(len(X_actual)):
    Xk_temp = torch.tensor(X_actual[i]) 
    phi_x_temp = phi_x(Xk_temp)
    Kx_temp    = K_net(phi_x_temp)
    y_kp1_temp = Kx_temp*phi_x_temp


    x_pred_temp = phi_d_x(y_kp1_temp)
    X_pred.append(x_pred_temp.tolist())


X_pred_plot = np.transpose(X_pred).tolist() 
Y_Actual    = np.transpose(Y_actual).tolist() 


x1p = X_pred_plot[1] 
x0p = X_pred_plot[0] 
x1a = Y_Actual[1]
x0a = Y_Actual[0] 
Ta = np.arange(0,len(x1a))/100
Tp = np.arange(0,len(x1p))/100

plt.figure(figsize=(9,6))
plt.subplot(2,1,1)
plt.plot(Tp,x0p,color="crimson",label="x1_predicted",linewidth=2.0)
plt.plot(Ta,x0a,color="blue",linestyle="--",label="x1_actual") 
plt.ylabel("x1") 
plt.title("Koopman Neural Network Embedding")
plt.legend(loc="best")
plt.grid()
plt.subplot(2,1,2)
plt.plot(Tp,x1p,color="teal",label="x2_predicted",linewidth=2.0)
plt.plot(Ta,x1a,color="orange",linestyle="--",label="x2_actual")
plt.ylabel("x2")
plt.xlabel("Time (seconds)")
plt.legend(loc="best")
plt.grid()
plt.savefig("Koopman_NN_result.jpg",dpi=420)
plt.show()
##################################################
##################################################







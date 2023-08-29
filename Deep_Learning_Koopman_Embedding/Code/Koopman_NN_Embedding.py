import os 
import torch 
import numpy as np 
import torch.nn as nn
import pandas as pd 
from torch.autograd import Variable
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


def Train_Koopman_Embedding(No_Episodes,X,Xd):
    phi_x       = Encoder()
    phi_d_x     = Decoder() 
    K_net       = Koopman_Eigen_Matrix()
    optimizer_0   = torch.optim.Adam(phi_x.parameters(),lr=1e-3) 
    optimizer_2   = torch.optim.Adam(phi_d_x.parameters(),lr=1e-3) 
    optimizer_1   = torch.optim.Adam(K_net.parameters(),lr=1e-3) 
    Koopman_net_loss        = list()
    for i in range(No_Episodes):
        if (i%100) == 0:
            print("Episode : ", i)
        #############################
        with torch.no_grad():
            yk            = phi_x(X)
            Lambda_K = K_net(yk) 
            y_kp1_    = Lambda_K*yk 
            
        y_kp1          = Variable(y_kp1_, requires_grad=False)
        x_kp1          = phi_d_x(y_kp1) 
        loss_phi_d_x   = torch.mean((Xd - x_kp1)**2)           # loss of decoder network
        optimizer_2.zero_grad()
        loss_phi_d_x.backward(retain_graph=True)
        optimizer_2.step()
        ############################
        Lambda_K = K_net(yk) 
        with torch.no_grad():
            K_phi_x_ref     = phi_x(Xd) 
        y_kp1    = Lambda_K*yk 
        loss_K          = torch.mean((K_phi_x_ref - y_kp1)**2)  # loss of koopman network
        optimizer_1.zero_grad() 
        loss_K.backward(retain_graph=True)
        optimizer_1.step() 
        ##############################
        yk            = phi_x(X)
        with torch.no_grad():
            pp_x_ref_      = phi_d_x(yk)

        pp_x_ref = Variable(pp_x_ref_, requires_grad=True)
        loss_Phi_x    = torch.mean((X - pp_x_ref)**2)           # loss of encoder network
        optimizer_0.zero_grad()   
        loss_Phi_x.backward(retain_graph=True) 
        optimizer_0.step()   
        #######################
        loss = loss_phi_d_x + loss_K + loss_Phi_x
        Koopman_net_loss.append(loss.item())
       ######################


    torch.save(phi_x.state_dict(), "Encoder.pt")  
    torch.save(phi_d_x.state_dict(), "Decoder.pt") 
    torch.save(K_net.state_dict(), "Koopman_EV.pt")  
    return Koopman_net_loss 


#############################################
################# Training ##################
#############################################
Xl = np.load("Xn.npy").tolist()
Yl = np.load("Yn.npy").tolist()

X,X_dash    = torch.tensor(Xl),torch.tensor(Yl)

Koopman_loss = Train_Koopman_Embedding(3000,X,X_dash)

plt.figure(figsize=(9,6))
plt.plot(Koopman_loss,color= "teal",linewidth=2.0)
plt.xlabel("Epochs") 
plt.ylabel("Mean Squared Error") 
plt.grid()
plt.title("Training Losses") 
plt.savefig("Koopman_embedding_loss.jpg",dpi=420) 
plt.show() 
###############################################
###############################################
###############################################
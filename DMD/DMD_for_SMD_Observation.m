%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Data Preparation %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MSD_Data = readtable("MSD_Obs_Data.csv");

X1 = MSD_Data.X1(1:2499,:);
X2 = MSD_Data.X2(1:2499,:);
X_raw = [X1 X2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Dynamic Mode Decomposition %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X          = X_raw(1:2498,:)';
X_dash     = X_raw(2:2499,:)';
%%%% singular value decomposition
[U,S,V] = svd( X ,"econ"); 
%%%% Approximation to the matrix A (A_tilde)
A_til   = (X_dash)*V*(S\U');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Result Checking %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yt_predicted = A_til*X;
Yt_actual    = X_dash;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Np = 2500;
DM = zeros(Np,2);
DM(1,:) = [2.5 0];

for i=1:Np-1
    temp = A_til*DM(i,:)';
    DM(i+1,:) = temp'; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% result plotting
T = (1:2498)*0.01;
plot(DM(:,1));
plot(T,Yt_actual(1,:),Color='#A2142F',LineWidth=2.5)
hold on;
plot(T,yt_predicted(1,:),Color='#77AC30',LineStyle='--',LineWidth=2.5)
legend({'Actual','Predicted'},'Location','southwest')
legend({},'Location','southwest')
title('Dynamic Mode Decomposition (DMD)');
xlabel('time (in seconds)');
ylabel('Ampliture (m)');
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- need to investigate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% further
%%%% Spatial Temporal mdoes
[W, eigs] = eig(A_til);
Phi = X_dash*V*inv(S)*W;
% 
% eigen = real(eigs); 
% 
% DM_phi = zeros(2,Np);
% 
% for i=1:2500
%     temp = real(Phi)*mpower(eigen,i)*[2.5; 0];
%     DM_phi(:,i) = temp;
% end 
% 
% plot(DM_phi(1,:))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




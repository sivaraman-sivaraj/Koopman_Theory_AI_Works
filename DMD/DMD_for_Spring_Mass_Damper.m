%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Data Preparation %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MSD_Data = readtable("MSD_sine_input_01.csv");

X1 = MSD_Data.X1(1:2499,:);
X2 = MSD_Data.X2(1:2499,:);
X_raw = [X1 X2];

U_raw = MSD_Data.U(2:2500);
%%% system properties
m = 5;
B = [0; 1/m];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Dynamic Mode Decomposition %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X          = X_raw(1:2498,:)';
X_dash     = X_raw(2:2499,:)';
Delta      = U_raw(1:2498,:)';
%%%% singular value decomposition
[U,S,V] = svd( X ,"econ"); 
%%%% Control gain matrix calculation
B_D = B*Delta;
%%%% Approximation to the matrix A (A_tilde)
A_til   = (X_dash- B_D)*V*(S\U')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% result checking 

TD = readtable("MSD_sine_input_01.csv");
X1_tst = TD.X1(1:2499,:);
X2_tst = TD.X2(1:2499,:);
X_raw_tst = [X1_tst X2_tst];

U_raw_tst = TD.U(2:2500);

X          = X_raw_tst(1:2498,:)';
X_dash     = X_raw_tst(2:2499,:)';
Delta      = U_raw_tst(1:2498,:)';

%%%% with train data
Xt           = X;
Ut           = Delta;
yt_predicted = A_til*Xt + (B*Ut);
Yt_actual    = X_dash;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% result plotting
T = (1:2498)*0.01;
plot(T,Yt_actual(1,:),Color='#A2142F',LineWidth=2.5)
hold on;
plot(T,yt_predicted(1,:),Color='#77AC30',LineStyle='--',LineWidth=2.5)
legend({'Actual','Predicted'},'Location','southwest')
% legend({},'Location','southwest')
title('Dynamic Mode Decomposition (DMD)');
xlabel('time (in seconds)');
ylabel('Ampliture (m)');
grid on;
















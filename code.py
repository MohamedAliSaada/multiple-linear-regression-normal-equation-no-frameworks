import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pprint import pprint 

#load the data and make needed changes to it .
df = pd.read_csv('ex_2.csv')
col = list(df.columns)
df= df[[col[-1]]+[i for i in col if i != col[-1]]]     

#make target and features in  vectorization form 
X = np.array(df.iloc[:,1:])
Y =np.array(df.iloc[:,-1]).reshape(-1,1)

# add bias term to X vector to use normal equation (solution[b W_vector] = (Xᵀ X)⁻¹ Xᵀ y )
X_pbias = np.c_[np.ones((X.shape[0])).reshape(-1,1),X]

#check size of vectoes and matrix
#print("X shape:", X.shape)
#print("Y shape:", Y.shape)
#print("X_pbias shape:", X_pbias.shape)

#solution using normal equation 
solution_vector = np.linalg.inv(X_pbias.T @ X_pbias) @ X_pbias.T @ Y

#the predicted 
Y_pred =X_pbias @ solution_vector

#add weights of my module in Data frame.
weights_x_names = [i for i in col if i != col[-1]]
head_of_weights = ['bias'] + [f"{i}_weight" for i in weights_x_names ]
data_of_weights = solution_vector.ravel()
solution_df =pd.DataFrame( { 'feature' :head_of_weights   , 'Value':    data_of_weights    }    )
solution_df.head()

#add my y_pred and actual y in data frame
com_df = pd.DataFrame({'Y_real':Y.ravel(),'Y_pred':Y_pred.ravel()})

#draw the comparison
plt.figure()
plt.scatter(Y,Y_pred,color='k',label='predictions')
plt.xlabel('Y_real')
plt.ylabel('Y_pred')
plt.title('Y_pred VS Y_real using normal equation !')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linestyle='--', label='Perfect Fit')
plt.grid(True)
plt.legend()
plt.show()
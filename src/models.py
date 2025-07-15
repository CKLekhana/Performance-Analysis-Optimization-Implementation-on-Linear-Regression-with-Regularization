# importing libraries
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse

# Creating a base linear regression class

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.losses = None
    
    def fit(self, X, y): # This class overridden in the best classes. It takes input X and target y to determine the intercept & coefficients
        pass
    
    def predict(self, X): # This class is used to predict the target given an input
        predicted_y = np.dot(X, self.coefficients) + self.intercept
        return predicted_y
    

# Ordinary Least Squares Implementation
class OLS(LinearRegression):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y): 
        X = np.insert(X, 0, 1, axis = 1) # inserts the intercept column in the input 
        XTX_inv = np.linalg.inv(np.dot(np.transpose(X), X))
        XTy = np.dot(np.transpose(X), y)
        theta = np.dot(XTX_inv, XTy).reshape(-1, )
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        self.losses = mse(y, np.dot(X, theta))
        print(f"Final Train Loss: {self.losses:.4f}")
        
    
# Ordinary Least Squares + Stochastic Gradient Descent Implementaion
class OLS_GD(LinearRegression):
    def __init__(self):
        super().__init__()
        self.epochs = None
        self.lr = None
        
    def fit(self, X, y, epochs=1000, lr=0.01, show_mse=False):
        self.epochs = epochs
        self.lr = lr
        self.losses = []
        n_samples, n_features = X.shape
        
        theta = np.zeros(n_features + 1)
        X = np.insert(X, 0, 1, axis = 1)
        
        for i in range(epochs):
            y_predicted = np.dot(X, theta)
            error = y - y_predicted
            gradient = (-2 / n_samples) * np.dot(np.transpose(X), error)
            theta = theta - lr * gradient
            epoch_mse = np.mean(error ** 2)
            self.losses.append(epoch_mse)
            if show_mse:
                print(f"Epoch {i+1} : MSE = {epoch_mse:.4f}")
            
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        final_mse = np.mean((y - np.dot(X, theta))**2)
        self.losses.append(final_mse)
        print(f"Final train loss = {final_mse:.4f}")
        
        
class Ridge(LinearRegression):
    
    def __init__(self):
        super().__init__()
        self.l2_param = None
        
    def fit(self, X, y, l2_param=0.01):
        
        self.l2_param = l2_param
        X = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X.shape
        
        identity_matrix = np.identity(n_features)
        identity_matrix[0, 0] = 0 # prevent regularization of intercept
        
        XTX = np.dot(np.transpose(X), X)
        XTX_lI = XTX + l2_param * identity_matrix
        XTX_lI_I = np.linalg.inv(XTX_lI)
        XT_y = np.dot(np.transpose(X), y)
        theta = np.dot(XTX_lI_I, XT_y)
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        final_mse = np.mean((y - np.dot(X, theta))**2)
        self.losses = final_mse
        print(f"Final Train Loss: {final_mse:.4f}")
        
    

class Ridge_GD(LinearRegression):
    
    def __init__(self):
        super().__init__()
        self.l2_param = None
        self.lr = None
        self.epochs = None
        
    def fit(self, X, y, epochs = 1000, lr = 0.01, l2_param = 0.01, show_mse = False):
        self.epochs = epochs
        self.lr = lr
        self.l2_param = l2_param
        self.losses = []
        n_samples, n_features = X.shape
        
        theta = np.zeros(n_features + 1)
        X = np.insert(X, 0, 1, axis = 1)
        
        for i in range(epochs):
            y_predicted = np.dot(X, theta)
            error = y - y_predicted
            coefficient_gradient = (-2 / n_samples) * (np.dot(np.transpose(X[ :, 1:]), error)) 
            intercept_gradient = (-2 / n_samples) * (np.dot(np.transpose(X[ :, 0]), error)) 
            theta[1:] = theta[1:] - lr * coefficient_gradient
            theta[0] = theta[0] - lr * intercept_gradient
            epoch_mse = np.mean(error ** 2)
            self.losses.append(epoch_mse)
            if show_mse:
                print(f"Epoch {i+1} : MSE = {epoch_mse:.4f}")
            
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        final_mse = np.mean((y - np.dot(X, theta))**2)
        self.losses.append(final_mse)
        print(f"Final train loss = {final_mse:.4f}")
        
    
class Lasso_CD(LinearRegression):
    
    def __init__(self):
        super().__init__()
        self.epochs = None
        self.l1_param = None
        
    def fit(self, X, y, epochs=1000, l1_param=0.01, show_mse = False):
        self.epochs = epochs
        self.l1_param = l1_param
        X = np.insert(X, 0, 1, axis =1 )
        n_samples, n_features = X.shape
        self.losses = []
        theta = np.zeros(n_features)
        residuals = y - np.dot(X, theta)
        for i in range(1, epochs +1):
            for j in range(n_features):
                Xj = X[:, j]
                residuals = residuals + np.dot(Xj, theta[j])
                if j == 0:
                    theta[j] = np.dot(Xj, residuals) / np.sum(Xj ** 2)
                else:
                    rho = np.dot(Xj, residuals)
                    theta[j] = self.soft_threshold(rho) / np.sum(Xj ** 2)
                residuals = residuals - (Xj * theta[j])
            self.losses.append(mse(y, np.dot(X, theta)))
            if show_mse:
                print(f"Epoch {i+1}: MSE = {self.losses[-1]:.4f}")
            
        
        self.coefficients = theta[1:]
        self.intercept = theta[0]
        predicted_y = np.dot(X, theta)
        self.losses.append(mse(y, predicted_y))
        print(f"Final Train Loss : {self.losses[-1]:.4f}")
               
    def soft_threshold(self, rho):
        if rho < -self.l1_param:
            return rho + self.l1_param
        elif rho > self.l1_param:
            return rho - self.l1_param
        else:
            return 0
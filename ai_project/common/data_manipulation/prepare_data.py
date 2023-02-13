from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(x: np.array):
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    return x
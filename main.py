import numpy as np
from ai_project.models.support_vector_regression.svr import SVR
from ai_project.common.data_manipulation.prepare_data import prepare_data
import pandas as pd
import os


def run_svr():
    path = os.path.join('ai_project', 'data', 'processed', 'house_price.csv')

    data = pd.read_csv(path)
    
    x = data.iloc[:, 1:].to_numpy()
    
    x = prepare_data(x)
    y = data['SalePrice'].to_numpy()
    
    x_train = x[:-100, :]
    y_train = y[:-100]
    
    x_test = x[-100:, :]
    y_test = y[-100:]

    model = SVR(epsilon=10000, loss_function='linear', kernel='quadratic')
    model.fit(x_train, y_train, n_iters=500, learning_rate=5000)
    predicted = [model.predict(np.array([x]))[0][0] for x in x_test]
    df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})
    df = df.round()
    df['Substract'] = df['Predicted'] - df['Actual']
    rmse = ((df['Predicted'] - df['Actual']) ** 2).mean() ** 0.5
    mae = ((df['Predicted'] - df['Actual']).abs()).mean()
    df.head(10)

    print(f'\n\nRMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    
    
if __name__ == '__main__':
    run_svr()
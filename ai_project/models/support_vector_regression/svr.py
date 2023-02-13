import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from ai_project.models.support_vector_regression.definitions.kernels import Kernels
from ai_project.models.support_vector_regression.definitions.loss_functions import LossFunctions
import numpy as np

tf.compat.v1.disable_eager_execution()

class SVR():
    def __init__(
        self,
        epsilon=0.5,
        loss_function: LossFunctions = 'linear',
        kernel: Kernels = 'linear'
        ):
        self.epsilon = epsilon
        self.loss_function = loss_function
        self.kernel = kernel
        
    def fit(self, X: np.array, y, n_iters=500, learning_rate=0.01):
        self.sess = tf.compat.v1.Session()
        
        if self.kernel == Kernels.LINEAR.value:
            feature_len = X.shape[-1] if len(X.shape) > 1 else 1
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            self.X = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, feature_len))
            self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1))
            
            self.W = tf.compat.v1.Variable(tf.random.normal(shape=(feature_len, 1)))
            self.b = tf.compat.v1.Variable(tf.random.normal(shape=(1,)))
        else:                
            feature_len = X.shape[-1] if len(X.shape) > 1 else 1
        
            df = pd.DataFrame(columns=range(0, feature_len), data=X)
            for col in range(feature_len):
                df[str(col)+'**2'] = df[col]** 2
            
            X = df.to_numpy()
            
            feature_len = X.shape[-1] if len(X.shape) > 1 else 1
            
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            self.X = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, feature_len))
            self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1))
            
            self.W = tf.compat.v1.Variable(tf.random.normal(shape=(feature_len, 1)))
            self.b = tf.compat.v1.Variable(tf.random.normal(shape=(1,)))
        
        self.y_pred = self._linear_kernel()
        
        self.loss = self._get_loss_function()

        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt_op = opt.minimize(self.loss)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        
        for i in range(n_iters):
            loss = self.sess.run(
                self.loss, 
                {
                    self.X: X,
                    self.y: y
                }
            )
            print("{}/{}: loss: {:.4f}".format(i + 1, n_iters, loss))
            
            self.sess.run(
                opt_op, 
                {
                    self.X: X,
                    self.y: y
                }
            )
            
        return self
            
    def predict(self, X, y=None):
        if self.kernel == Kernels.QUADRATIC.value:
            feature_len = X.shape[-1] if len(X.shape) > 1 else 1
        
            df = pd.DataFrame(columns=range(0, feature_len), data=X)
            for col in range(feature_len):
                df[str(col)+'**2'] = df[col]** 2
            
            X = df.to_numpy()
            
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        y_pred = self.sess.run(
            self.y_pred, 
            {
                self.X: X 
            }
        )
        return y_pred

        
    def plot_result(self, x, y):
        plt.plot(
            x, self.predict(x), "-"
        )
        
        plt.plot(
            x, self.predict(x) + self.epsilon, "--", color='black'
        )
        
        plt.plot(
            x, self.predict(x) - self.epsilon, "--", color='black'
        )
        
        plt.legend(["actual", "prediction"])
        plt.show()
    
    def _get_loss_function(self):
        if self.loss_function == LossFunctions.LINEAR.value:
            return self._linear_loss_function()
        elif self.loss_function == LossFunctions.QUADRATIC.value:
            return self._quadratic_loss_function()
        elif self.loss_function == LossFunctions.HUBER.value:
            return self._huber_loss_function()
        else:
            raise(ValueError(f'Acceptable loss functions are: {LossFunctions._member_names_}'))
        
    def _linear_kernel(self):
        return tf.matmul(self.X, self.W) + self.b
            
    def _linear_loss_function(self):
        return tf.reduce_mean(tf.maximum(0., tf.abs(self.y - self.y_pred) - self.epsilon))
    
    def _quadratic_loss_function(self):
        return self._linear_loss_function() ** 2
    
    def _huber_loss_function(self):
        return tf.cond(
            tf.reduce_mean(tf.abs(self.y - self.y_pred)) > self.epsilon,
            lambda: tf.reduce_mean(self.epsilon * tf.abs(self.y - self.y_pred) - self.epsilon ** 2 / 2),
            lambda: tf.reduce_mean(1 / 2  * tf.abs(self.y - self.y_pred)) ** 2)

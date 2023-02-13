from enum import Enum

class LossFunctions(Enum):
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'
    HUBER = 'huber'
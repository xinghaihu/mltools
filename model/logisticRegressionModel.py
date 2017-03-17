import numpy as np

import random

'''
__author__  = 'Xinghai Hu'
'''

class logisticRegressionModel:
    def __init__(self, featNum, weightVec = None, withInterceptOrNot = True, intercept = 0.0, randomInit = True, \
                 seed = -1):
        self.featNum = featNum
        if weightVec == None:
            if randomInit:
                self.weightVec = np.random.rand(featNum)
                self.intercept = float(random.random)
                if seed >= 0:
                    np.random.seed(seed)
                    self.weightVec = np.random.rand(featNum)
                    random.seed(seed)
                    self.intercept = float(random.random)
            else:
                self.weightVec = np.zeros(featNum)
                self.intercept = 0.0
        else:
            if weightVec.size != featNum:
                raise ValueError("input argument weightVec length does not match argument featNum")
            else:
                self.weightVec = weightVec
                self.intercept = intercept
        self.withInterceptOrNot = withInterceptOrNot
        if self.withInterceptOrNot:
            self.modelVec = np.append(self.weightVec, self.intercept)
        else:
            self.modelVec = self.weightVec

    def training(self, trainDF, method = 'gradient_desc', learningParams = {'learningRate' : .5, \
                                                                            'maxIter' : 100, \
                                                                            'minStep' : .001, \
                                                                            }):
        if method == 'gradient_desc':
            iter = 0
            while iter < learningParams['maxIter']:
                grad = computeGrad(self.modelVec, trainDF)
                self.modelVec = self.modelVec - grad * learningParams['learningRate']
                if np.linalg.norm(grad) < learningParams['minStep']:
                    break

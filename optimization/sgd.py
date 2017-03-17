import numpy as np

'''
__author__  Xinghai Hu
'''

class sgd:
    '''
    This class implements an optimization plugin using Stochastic Gradient Descent (SGD).
    It supports:
        * initialization (random initialization or with fixed solution)
        * mini batch
        * momentum -- support two types of momentum:
            ** Flat momentum: m_t <- (1-lambda)*m_{t-1} + lambda*g_t, w_{t+1} <- w_t - ita_t*g_t - ita_t'*m_t
            ** Nesterov momentum: m_{t+1} <- mu*m_t + epsilon*g(w_t-mu*m_t), w_{t+1} <- w_t - m_{t+1}
        * adaptive learning rate -- support the following learning rates:
            ** polynomial decay: ita(t) <- alpha / (beta + t)^gamma. gamma by default uses .75 (used by Bottou), \
                conventional SGD uses .5 and strong convexity uses 1.0
            ** Adagrad: the learning rate is adaptive to the magnitude of computed gradient. \
                adjusted gradient equals to original gradient divided by square root of running average of historical \
                gradients plus fudge factor (to guarantee numeric stability)
            ** RMSProp:
            ** ADAM (adaptive moment estimation):
        * early stop and convergence control

    References:
        * Alex Smola's lecture videos in CMU: https://www.youtube.com/watch?v=Zm8l-JQAJD8
        * http://sebastianruder.com/optimizing-gradient-descent/
        * https://en.wikipedia.org/wiki/Stochastic_gradient_descent
        * Jeffery Hinton lecture notes in UToronto: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    '''
    def __init__(self, initSolution, batchSize = 128, momentum = "None", momentum_params = {}, \
                 initialLearningRate = .1, ):
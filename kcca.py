import numpy
import scipy.linalg

class GaussianKernel(object):
    """
    A class to find gaussian kernel evaluations k(x, y) = exp (-||x - y||^2/2 sigma^2)
    """
    def __init__(self, sigma=1.0):
        """
        Initialise object with given value of sigma >= 0

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        """
        Find kernel evaluation between two matrices X1 and X2 whose rows are
        examples and have an identical number of columns.


        :param X1: First set of examples.
        :type X1: :class:`numpy.ndarray`

        :param X2: Second set of examples.
        :type X2: :class:`numpy.ndarray`
        """
        
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        j1 = numpy.ones((X1.shape[0], 1))
        j2 = numpy.ones((X2.shape[0], 1))

        diagK1 = numpy.sum(X1**2, 1)
        diagK2 = numpy.sum(X2**2, 1)

        X1X2 = numpy.dot(X1, X2.T)

        Q = (2*X1X2 - numpy.outer(diagK1, j2) - numpy.outer(j1, diagK2) )/ (2*self.sigma**2)

        return numpy.exp(Q)

    def __str__(self):
        return "GaussianKernel: sigma = " + str(self.sigma)

class LinearKernel(object):
    """
    A class to find linear kernel evaluations k(x, y) = <x, y> 
    """
    def __init__(self):
        """
        Intialise class. 
        """
        pass

    def evaluate(self, X1, X2):
        """
        Find kernel evaluation between two matrices X1 and X2 whose rows are
        examples and have an identical number of columns.


        :param X1: First set of examples.
        :type X1: :class:`numpy.ndarray`

        :param X2: Second set of examples.
        :type X2: :class:`numpy.ndarray`
        """

        if X1.shape[1] != X2.shape[1]:
            raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        return numpy.dot(X1, X2.T)
    
class KCCA(object):
    def __init__(self, kernel1, kernel2, regularization, method = 'kettering_method'):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.reg = regularization
        self.method = getattr(self, method)
        
        self.alpha1 = None
        self.alpha2 = None
        self.trainX1 = None
        self.trainX2 = None

    def standard_hardoon_method(self, K1, K2, reg):
        
        N = K1.shape[0]
        I = numpy.eye(N)
        Z = numpy.zeros((N,N))    
        
        R1 = numpy.c_[Z, numpy.dot(K1, K2)]
        R2 = numpy.c_[numpy.dot(K2, K1), Z]
        R =  numpy.r_[R1, R2]
        
        D1 = numpy.c_[numpy.dot(K1, K1 + reg*I), Z]
        D2 = numpy.c_[Z, numpy.dot(K2, K2 + reg*I)]
        D = 0.5*numpy.r_[D1, D2]
        
        return (R, D)    
            
    def simplified_hardoon_method(self, K1, K2, reg):
        
        N = K1.shape[0]
        I = numpy.eye(N)
        Z = numpy.zeros((N,N))    
        
        R1 = numpy.c_[Z, K2]
        R2 = numpy.c_[K1, Z]
        R =  numpy.r_[R1, R2]
        
        D1 = numpy.c_[K1 + reg*I, Z]
        D2 = numpy.c_[Z, K2 + reg*I]
        D = 0.5*numpy.r_[D1, D2]
        
        return (R, D)    
            
    def kettering_method(self, K1, K2, reg):
        
        N = K1.shape[0]
        I = numpy.eye(N)
        Z = numpy.zeros((N,N))    
        
        R1 = numpy.c_[K1, K2]
        R2 = R1
        R = 1./2 * numpy.r_[R1, R2]
        
        D1 = numpy.c_[K1 + reg*I, Z]
        D2 = numpy.c_[Z, K2 + reg*I]
        D = numpy.r_[D1, D2]
        
        return (R, D)        

    def kcca(self, K1, K2):
        
        N = K1.shape[0]
        N0 = numpy.eye(N) - 1./N * numpy.ones(N)
        K1 = numpy.dot(numpy.dot(N0,K1),N0)
        K2 = numpy.dot(numpy.dot(N0,K2),N0)    
        R, D = self.method(K1, K2, self.reg)
        
        #solve generalized eigenvalues problem
        betas, alphas = scipy.linalg.eig(R,D)
        ind = numpy.argsort(numpy.real(betas))
        max_ind = ind[-1]
        alpha = alphas[:, max_ind]
        alpha = alpha/numpy.linalg.norm(alpha)
        beta = numpy.real(betas[max_ind])
        
        alpha1 = alpha[:N]
        alpha2 = alpha[N:]
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        y1 = numpy.dot(K1, alpha1)
        y2 = numpy.dot(K2, alpha2)
        
        return (y1, y2, beta)        
        
    def fit(self, X1, X2):
        self.trainX1 = X1
        self.trainX2 = X2
        
        K1 = self.kernel1(X1, X1)
        K2 = self.kernel2(X2, X2)
        
        self.kcca(K1, K2)
        return self
    
    def transform(self, X1 = None, X2 = None):
        rets = []
        if X1 is not None:
            testTrainK1 = self.kernel1(X1, self.trainX1)
            rets.append(numpy.dot(testTrainK1, self.alpha1))
            
        if X2 is not None:
            testTrainK2 = self.kernel1(X2, self.trainX2)
            rets.append(numpy.dot(testTrainK2, self.alpha2))
            
        return rets
    
            
            
            
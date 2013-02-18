import numpy
from numpy import dot, eye, ones, zeros
import scipy.linalg

from kernel_icd import kernel_icd


    
class KCCA(object):
    """An implementation of Kernel Canonical Correlation Analysis. 
    
    """
    def __init__(self, kernel1, kernel2, regularization, method = 'kettering_method',
                 decomp = 'full', lrank = None):

        if decomp not in ('full', 'icd'):
            raise ValueError("Error: valid decom values are full or icd, received: "+str(decomp))
        
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.reg = regularization
        self.method = getattr(self, decomp + "_" + method)
        
        self.decomp = decomp
        self.lrank = lrank
        
        self.alpha1 = None
        self.alpha2 = None
        self.trainX1 = None
        self.trainX2 = None
        

    def full_standard_hardoon_method(self, K1, K2, reg):
        
        N = K1.shape[0]
        I = eye(N)
        Z = numpy.zeros((N,N))    
        
        R1 = numpy.c_[Z, dot(K1, K2)]
        R2 = numpy.c_[dot(K2, K1), Z]
        R =  numpy.r_[R1, R2]
        
        D1 = numpy.c_[dot(K1, K1 + reg*I), Z]
        D2 = numpy.c_[Z, dot(K2, K2 + reg*I)]
        D = 0.5*numpy.r_[D1, D2]
        
        return (R, D)    
            
    def full_simplified_hardoon_method(self, K1, K2, reg):
        
        N = K1.shape[0]
        I = eye(N)
        Z = numpy.zeros((N,N))    
        
        R1 = numpy.c_[Z, K2]
        R2 = numpy.c_[K1, Z]
        R =  numpy.r_[R1, R2]
        
        D1 = numpy.c_[K1 + reg*I, Z]
        D2 = numpy.c_[Z, K2 + reg*I]
        D = numpy.r_[D1, D2]
        
        return (R, D)    
            
    def full_kettering_method(self, K1, K2, reg):
        
        N = K1.shape[0]
        I = eye(N)
        Z = numpy.zeros((N,N))    
        
        R1 = numpy.c_[K1, K2]
        R2 = R1
        R = 1./2 * numpy.r_[R1, R2]
        
        D1 = numpy.c_[K1 + reg*I, Z]
        D2 = numpy.c_[Z, K2 + reg*I]
        D = numpy.r_[D1, D2]
        
        return (R, D)        

    def kcca(self, K1, K2):
        
        #remove the mean in features space
        N = K1.shape[0]
        N0 = eye(N) - 1./N * ones(N)
        K1 = dot(dot(N0,K1),N0)
        K2 = dot(dot(N0,K2),N0)
        
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
        
        y1 = dot(K1, alpha1)
        y2 = dot(K2, alpha2)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        return (y1, y2, beta)        

    def icd_simplified_hardoon_method(self, G1, G2, reg):
        N1 = G1.shape[1]
        N2 = G2.shape[1]
        
        Z11 = zeros((N1, N1))
        Z22 = zeros((N2, N2))
        Z12 = zeros((N1,N2))
        
        I11 = eye(N1)
        I22 = eye(N2)

        R1 = numpy.c_[Z11, dot(G1.T, G2)]
        R2 = numpy.c_[dot(G2.T, G1), Z22]
        R =  numpy.r_[R1, R2]
        
        D1 = numpy.c_[dot(G1.T, G1) + reg*I11, Z12]
        D2 = numpy.c_[Z12.T, dot(G2.T, G2) + reg*I22]
        D = numpy.r_[D1, D2]
        
        return (R, D)            
    
    def icd(self, G1, G2):
        """Incomplete Cholesky decomposition
        """
        
        # remove mean. avoid standard calculation N0 = eye(N)-1/N*ones(N);
        G1 = G1 - numpy.array(numpy.mean(G1, 0), ndmin=2, copy=False)
        G2 = G2 - numpy.array(numpy.mean(G2, 0), ndmin=2, copy=False)

        R, D = self.method(G1, G2, self.reg)
        
        #solve generalized eigenvalues problem
        betas, alphas = scipy.linalg.eig(R,D)
        ind = numpy.argsort(numpy.real(betas))
        max_ind = ind[-1]
        alpha = alphas[:, max_ind]
        alpha = alpha/numpy.linalg.norm(alpha)
        beta = numpy.real(betas[max_ind])
        
        N1 = G1.shape[1]
        alpha1 = alpha[:N1]
        alpha2 = alpha[N1:]
        
        y1 = dot(G1, alpha1)
        y2 = dot(G2, alpha2)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        return (y1, y2, beta)                
    
    def fit(self, X1, X2):
        self.trainX1 = X1
        self.trainX2 = X2
        
        if self.decomp == "full":
            self.K1 = self.kernel1(X1, X1)
            self.K2 = self.kernel2(X2, X2)            
            (y1, y2, beta) = self.kcca(self.K1, self.K2)
        else:
            # get incompletely decomposed kernel matrices. K \approx G*G'
            self.K1 = kernel_icd(X1, self.kernel1,  self.lrank)
            self.K2 = kernel_icd(X2, self.kernel2,  self.lrank)            
            (y1, y2, beta) = self.icd(self.K1, self.K2)

        self.y1_ = y1
        self.y2_ = y2
        self.beta_ = beta
        return self
    
    def transform(self, X1 = None, X2 = None):
        """
        
        Features centering taken from:
        Scholkopf, B., Smola, A., & Muller, K. R. (1998).
        Nonlinear component analysis as a kernel eigenvalue problem.
        Neural computation, 10(5), 1299-1319.
        """
        rets = []
        if X1 is not None:
            Ktest = self.kernel1(X1, self.trainX1)            
            K = self.K1
            
            L, M = Ktest.shape
            ones_m = ones((M, M))
            ones_mp = ones((L, M)) / M
            
            #features centering
            K1 = (Ktest - dot(ones_mp, K)
                  - dot(Ktest, ones_m) + dot(dot(ones_mp, K), ones_m)
                  )
            
            res1 =  dot(K1, self.alpha1)
            rets.append(res1)
            
        if X2 is not None:
            Ktest = self.kernel2(X2, self.trainX2)            
            K = self.K2
            
            L, M = Ktest.shape
            ones_m = ones((M, M))
            ones_mp = ones((L, M)) / M

            #features centering
            K2 = (Ktest - dot(ones_mp, K)
                  - dot(Ktest, ones_m) + dot(dot(ones_mp, K), ones_m)
                  )
            
            res2 =  dot(K2, self.alpha2)
            rets.append(res2)
            
        return rets
    
if __name__ == "__main__":
    from kernels import DiagGaussianKernel
    x1 = numpy.array([0.0764, 0.6345, 0.1609, 0.0384, 0.8558], ndmin=2).T
    x2 = numpy.array([0.7273, 0.4829, 0.3440, 0.4406, 0.8074], ndmin=2).T
    kernel = DiagGaussianKernel()
    cca = KCCA(kernel, kernel,
                    regularization=1e-5,
                    decomp='icd',
                    method='simplified_hardoon_method').fit(x1,x2)
    
    print "Done"
    print cca.y1_
    print cca.y2_
    print cca.beta_
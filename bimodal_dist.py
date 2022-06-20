import numpy as np
import scipy as sp


def gauss(x, sig, mu):
    return np.exp(-(x-mu)**2/(2*sig**2))

def bimodal(x, std, mean):
    if x >= 0:
        return gauss(x, std, mean) 
    if x < 0:
     return np.exp(x)*gauss(-x, std, mean)
    
def loggauss(x, sig, mu):
    return -(x-mu)**2/(2*sig**2)

def log_bimodal(x, std, mean):
    if x >= 0:
        out = loggauss(x, std, mean) 
    if x < 0:
        out = x + loggauss(-x, std, mean)
    
    return np.exp(out)

def generate_dist(pdf_nonorm, args):
    pdf = lambda x: pdf_nonorm(x, *args)
    dist = ContinuousDist()
    dist.set_pdf(pdf)
    return dist

class ContinuousDist(sp.stats.rv_continuous):

    def set_pdf(self, pdf_func, lims=[-np.inf, np.inf]):
        self.norm = sp.integrate.quad(pdf_func, *lims)[0]
        self.pdf_func = pdf_func
        
        self._pdf = lambda y: self.pdf_func(y) / self.norm

    def get_min_eps(self):
        kernel = lambda x: np.tanh(x/2) * self.pdf(x)
        tanh_avg = sp.integrate.quad(kernel, -np.inf, np.inf)[0]
        return (1/tanh_avg) - 1
    def check_mean(self):
        kernel = lambda x: x * self.pdf(x)
        xmean = sp.integrate.quad(kernel, -np.inf, np.inf)[0]
        return xmean

    def check_var(self):
        mean = check_mean(self)
        kernel = lambda x: (x-mean)**2 * self.pdf(x)
        var = sp.integrate.quad(kernel, -np.inf, np.inf)[0]
        return var

    def check_mean_num(self, min=-30, max=40, N=200_000):
        x=np.linspace(min, max, N)
        dx=x[1]-x[0]
        pdf_x = [self.pdf(item) for item in x]
        return dx*sum(np.multiply(x,pdf_x))

    def check_mean_sample(self, N):
        return np.mean(self.rvs(size=N))
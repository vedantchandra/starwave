import pyabc
from collections import UserDict

class SWParameters(UserDict):
    """
    dict-like class that contains SWParameter objects
    
    Attributes
    ----------
    dict : dict
        dictionary of SWParameter objects
    
    Methods
    -------
    get_dict():
        Returns the base dictionary of SWParameter objects
    summary():
        Prints a summary of all contained parameters and their priors
    to_pyabc():
        Converts parameters into pyABC prior distributions
    
    """
    
    def __init__(self, params_dict):
        
        super().__init__(params_dict)
        self.dict = params_dict
        
    def __getitem__(self, param):
         return self.dict[param]
        
    def get_dict(self):
        return self.dict
    
    def summary(self):
        print_prior_summary(self.dict)
        
    def to_pyabc(self):
        return make_prior(self)

class SWParameter(UserDict):
    """
    dict-like class for each parameter's prior
    
    Attributes
    ----------
    name : dict
        parameter name
    value : float
        initial parameter value
    bounds : array-like
        lower and upper bounds on the parameter. ignored if distribution != 'uniform'
    distribution : str
        prior distribution name in scipy.stats naming convention
    dist_kwargs : dict
        arguments to specify non-uniform priors, for example 'mean' and 'sigma' if 
        distribution is 'norm'
    fixed : bool
        whether to fix or vary parameter during sampling
    
    Methods
    -------
    set(**attributes):
        Set any of the parameter's attributes. Any attribute not explicitly 
        set is left at the default. 
    
    """
    
    def __init__(self, name, value, bounds, distribution = 'uniform', dist_kwargs = None, fixed = False):
        self.name = name
        self.value = value
        self.bounds = bounds
        self.distribution = distribution
        self.dist_kwargs = dist_kwargs
        self.fixed = fixed
        
        self.param_dict = dict(name = self.name, value = self.value, bounds = self.bounds,
                          distribution = self.distribution, dist_kwargs = self.dist_kwargs,
                          fixed = self.fixed)
        
        super().__init__(self.param_dict)
        
    def set(self, value = None, bounds = None, distribution = None, dist_kwargs = None, fixed = None):
        
        if value is not None:
            self.value = value
        if bounds is not None:
            self.bounds = bounds
        if distribution is not None:
            self.distribution = distribution
        if dist_kwargs is not None:
            self.dist_kwargs = dist_kwargs
        if fixed is not None:
            self.fixed = fixed
            
        self.param_dict = dict(name = self.name, value = self.value, bounds = self.bounds,
                          distribution = self.distribution, dist_kwargs = self.dist_kwargs,
                          fixed = self.fixed)
        
        super().__init__(self.param_dict)
        
def make_params(imf_type):
    
    parameters = {};
    parameters['bf'] = SWParameter('bf', 0.2, [0, 1])
    parameters['log_int'] = SWParameter('log_int', 2, [2, 6])
    
    if imf_type == 'spl':
        
        parameters['slope'] = SWParameter('slope', -2.3, [-4, -1])
        
    
    elif imf_type == 'bpl':
        
        parameters['alow'] = SWParameter('alow', -1.3, [-2, 0])
        parameters['ahigh'] = SWParameter('ahigh', -2.3, [-4, -1])
        parameters['bm'] = SWParameter('bm', 0.5, [0.2, 0.8])

    elif imf_type == 'ln':
        
        parameters['mean'] = SWParameter('mean', 0.25, [0.1, 0.8])
        parameters['sigma'] = SWParameter('sigma', 0.6, [0.1, 1])
        parameters['bm'] = SWParameter('bm', 1, [0.8, 1.2])
        parameters['slope'] = SWParameter('slope', -2.3, [-3, -1])
        
    return SWParameters(parameters)


def make_prior(parameters):
    
    priors = {};
    
    for name, param in parameters.dict.items():
        
        if param.fixed:
            priors[name] = pyabc.RV(param.distribution, param.value, 0)
            continue
        
        
        lower = param.bounds[0]
        upper = param.bounds[1]
        
        if param.distribution == 'uniform':    
            priors[name] = pyabc.RV(param.distribution, lower, upper - lower)
            
        elif param.distribution == 'norm':
            try:
                mean = param.dist_kwargs['mean']
                sigma = param.dist_kwargs['sigma']
            except:
                raise ValueError('please pass valid distribution arguments!')
            
            priors[name] = pyabc.RV(param.distribution, mean, sigma)
            
        else:
            raise ValueError('invalid distribution name')
            
    pyabc_priors = pyabc.Distribution(**priors)
    
    return pyabc_priors

def print_prior_summary(parameters):
    
    for name, param in parameters.items():
        print('-'*10)
        print(name)
        print('-'*10)
        print('Distribution: ', end =" ")
        print(param.distribution)
        print('Bounds: ', end =" ")
        print(param.bounds)
        print('Value: ', end =" ")
        print(param.value)
        print('Fixed: ', end =" ")
        print(param.fixed)
        print('dist_kwargs: ', end =" ")
        print(param.dist_kwargs)

if __name__ == '__main__':
    p = make_params('bpl')
    print(make_prior(p))
    p.pretty_print()
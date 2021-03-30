import pyabc
from collections import OrderedDict
import sbi
from torch import Tensor, float32
from torch.distributions import Distribution
from typing import Optional, Union, Sequence
from scipy import stats
import torch
import copy

class SWDist:
    def __init__(self, distribution):
        self.dist = distribution # distribution is a scipy object

    def sample(self, N):
        return self.dist.rvs(N)

    def log_prob(self, x):
        return self.dist.logpdf(x)

class SWParameters(OrderedDict):
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
        
    def get_values(self):
        values_dict = {};
        for name, param in self.dict.items():
            values_dict[name] = param.value
            
        return values_dict
    
    def summary(self):
        print_prior_summary(self.dict)

    def to_torch(self):
        return make_prior(self)
        
    def to_pyabc(self):
        return make_prior_pyabc(self)

class SWParameter(OrderedDict):
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
        
def make_params(imf_type, sfh_type): # add SFH TYPE
    
    parameters = {};
    param_mapper = {};
    parameters['log_int'] = SWParameter('log_int', 2, [2, 6])
    parameters['bf'] = SWParameter('bf', 0.2, [0, 1])

    parameters['dm'] = SWParameter('dm', 0, [0, 1], fixed = True)
    parameters['sig_dm'] = SWParameter('sig_dm', 0.1, [0, 0.5], fixed = True)
    
    ## ADD EXTINCTION WITH EXTINCT PACKAGE    

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

    if sfh_type == 'gaussian':
        parameters['age'] = SWParameter('age', 5, [0.1, 13.4])
        parameters['sig_age'] = SWParameter('sig_age', 1, [0.1, 5])
        parameters['feh'] = SWParameter('feh', -1, [-4, 1])
        parameters['sig_feh'] = SWParameter('sig_feh', 0.1, [0.05, 1])
        parameters['age_feh_corr'] = SWParameter('age_feh_corr', -0.5, [-1, 0])

    # for ii,parameter in enumerate(parameters.keys()):
    #     param_mapper[ii] = parameter
    # print(param_mapper)
    return SWParameters(parameters)#, param_mapper

# def make_prior_pyabc(parameters):
    
#     priors = OrderedDict{};
    
#     for name, param in parameters.dict.items():
        
#         if param.fixed:
#             priors[name] = pyabc.RV(param.distribution, param.value, 0)
#             continue
        
        
#         lower = param.bounds[0]
#         upper = param.bounds[1]
        
#         if param.distribution == 'uniform':    
#             priors[name] = pyabc.RV(param.distribution, lower, upper - lower)
            
#         elif param.distribution == 'norm':
#             try:
#                 mean = param.dist_kwargs['mean']
#                 sigma = param.dist_kwargs['sigma']
#             except:
#                 raise ValueError('please pass valid distribution arguments!')
            
#             priors[name] = pyabc.RV(param.distribution, mean, sigma)
            
#         else:
#             raise ValueError('invalid distribution name')
            
#     pyabc_priors = pyabc.Distribution(**priors)
    
#     return pyabc_priors



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

class MultipleIndependent(Distribution):
    """Wrap a sequence of PyTorch distributions into a joint PyTorch distribution.

    Every element of the sequence is treated as independent from the other elements.
    Single elements can be multivariate with dependent dimensions, e.g.,:
        - [
            Gamma(torch.zeros(1), torch.ones(1)),
            Beta(torch.zeros(1), torch.ones(1)),
            MVG(torch.ones(2), torch.tensor([[1, .1], [.1, 1.]]))
        ]
        - [
            Uniform(torch.zeros(1), torch.ones(1)),
            Uniform(torch.ones(1), 2.0 * torch.ones(1))]    
    """

    def __init__(
        self, dists: Sequence[Distribution], validate_args=None,
    ):
        self._check_distributions(dists)

        self.dists = dists
        # numel() instead of event_shape because for all dists both is possible,
        # event_shape=[1] or batch_shape=[1]
        self.dims_per_dist = torch.as_tensor([d.sample().numel() for d in self.dists])
        self.ndims = torch.sum(torch.as_tensor(self.dims_per_dist)).item()

        super().__init__(
            batch_shape=torch.Size([]),  # batch size was ensured to be <= 1 above.
            event_shape=torch.Size(
                [self.ndims]
            ),  # Event shape is the sum of all ndims.
            validate_args=validate_args,
        )

    def _check_distributions(self, dists):
        """Check if dists is Sequence and longer 1 and check every member."""
        assert isinstance(
            dists, Sequence
        ), f"""The combination of independent priors must be of type Sequence, is 
               {type(dists)}."""
        assert len(dists) > 1, "Provide at least 2 distributions to combine."
        # Check every element of the sequence.
        [self._check_distribution(d) for d in dists]

    def _check_distribution(self, dist: Distribution):
        """Check type and shape of a single input distribution."""

        assert not isinstance(
            dist, MultipleIndependent
        ), "Nesting of combined distributions is not possible."
        assert isinstance(
            dist, Distribution
        ), "Distribution must be a PyTorch distribution."
        # Make sure batch shape is smaller or equal to 1.
        assert dist.batch_shape in (
            torch.Size([1]),
            torch.Size([0]),
            torch.Size([]),
        ), "The batch shape of every distribution must be smaller or equal to 1."

        assert (
            len(dist.batch_shape) > 0 or len(dist.event_shape) > 0
        ), """One of the distributions you passed is defined over a scalar only. Make
        sure pass distributions with one of event_shape or batch_shape > 0: For example
            - instead of Uniform(0.0, 1.0) pass Uniform(torch.zeros(1), torch.ones(1))
            - instead of Beta(1.0, 2.0) pass Beta(tensor([1.0]), tensor([2.0])).
        """

    def sample(self, sample_shape=torch.Size()) -> Tensor:

        # Sample from every sub distribution and concatenate samples.
        sample = torch.cat([d.sample(sample_shape) for d in self.dists], dim=-1)

        # This reshape is needed to cover the case .sample() vs. .sample((n, )).
        if sample_shape == torch.Size():
            sample = sample.reshape(self.ndims)
        else:
            sample = sample.reshape(-1, self.ndims)

        return sample

    def log_prob(self, value) -> Tensor:

        value = self._prepare_value(value)

        # Evaluate value per distribution, taking into account that individual
        # distributions can be multivariate.
        num_samples = value.shape[0]
        log_probs = []
        dims_covered = 0
        for idx, d in enumerate(self.dists):
            ndims = self.dims_per_dist[idx].item()
            v = value[:, dims_covered : dims_covered + ndims]
            # Reshape here to ensure all returned log_probs are 2D for concatenation.
            log_probs.append(d.log_prob(v).reshape(num_samples, 1))
            dims_covered += ndims

        # Sum accross last dimension to get joint log prob over all distributions.
        return torch.cat(log_probs, dim=1).sum(-1)

    def _prepare_value(self, value) -> Tensor:
        """Return input value with fixed shape.

        Raises: 
            AssertionError: if value has more than 2 dimensions or invalid size in
                2nd dimension.
        """

        if value.ndim < 2:
            value = value.unsqueeze(0)

        assert (
            value.ndim == 2
        ), f"value in log_prob must have ndim <= 2, it is {value.ndim}."

        batch_shape, num_value_dims = value.shape

        assert (
            num_value_dims == self.ndims
        ), f"Number of dimensions must match dimensions of this joint: {self.ndims}."

        return value

    @property
    def mean(self) -> Tensor:
        return torch.cat([d.mean for d in self.dists])

    @property
    def variance(self) -> Tensor:
        return torch.cat([d.variance for d in self.dists])


if __name__ == '__main__':
    p = make_params('bpl')
    print(make_prior(p))
    print_prior_summary(p)

import sys, os, re

import numpy as np

import matplotlib.pyplot as plt

from iminuit import Minuit, cost



def gauss(x, A, mu, sigma, index=2):
    '''
    Standard Gaussian function. Accepts scalars and numpy arrays
    as arguments for 'x', scalars for all other parameters.
    Includes an optional index to make flat-top Gaussian functions,
    which can sometimes be useful

    INPUTS

        x - single scalar, or 1D array-like. Independent variable
            i.e. the horizontal axis

        A - single scalar. The peak value, i.e. when x == mu, 
            gauss == A

        mu - single scalar. the mean value of the gaussian

        sigma - single scalar. the standard deviation

        index - optional, single scalar. Set the power law 
            dependence of the gaussian. Larger values make a 
            more flat-topped function

    OUTPUTS

        gauss = A * np.exp( -(x - mu)**index / (2.0 * sigma**index) )
    '''

    return A * np.exp( -(x - mu)**index / (2.0 * sigma**index) )



def fit_gaussian(x, y, y_unc=[], p0=[], poisson_unc=False, errordef=1.0, \
                 print_level=0, offset_prior=0.0, offset_prior_unc=np.inf, \
                 param_unc=False):
    '''
    Takes an input dataset and fits a Gaussian function. Can incorporate
    user-provided uncertainties, or use Poissonian uncertainties for bin 
    counts. In the absence of either, fitting weights are set to unity

    INPUTS

        x - 1D array of independent variable values

        y - 1D array of the depedent variable values

        y_unc - 1D array of uncertainties on dependent variables

        p0 - list or tuple containing initial values of parameters to 
            try. the function will try to find these by default, but if 
            you know them and want to provide them, by all means

        poisson_unc - boolean, True to use Poissonian uncertainies

        errordef - argument passed to 'Minuit()'. See that documentation.
            Sets the "Delta chi-squared", i.e. how far up the cost function
            you climb from the minimum to determine one standard deviation 
            of the inferred best parameter

        print_level - argument passed to 'Minuit()'. See that documentation.
            Integer from 0-3 with increasing levels of debug messages

        offset_prior - float, optional. if a constant offset is known, this
            parameter can constrain the fit to include it

        param_unc - boolean, specifies whether to use MINOS to compute
            uncertainties on the best-fit parameters

    OUTPUT

        popt = [A, mu, sigma] - array of best fit parameter values
    '''

    if not y_unc:
        if poisson_unc:
            y_unc = np.sqrt(y)
        else:
            y_unc = np.ones(len(y))

    ### Define least-squared cost function for fitting
    if offset_prior:
        wrapper = lambda x,A,mu,sigma,c: gauss(x,A,mu,sigma) + c
        costfunc = cost.LeastSquares(x, y, y_unc, wrapper) + \
            cost.NormalConstraint('c', offset_prior, offset_prior_unc)
    else:
        wrapper = lambda x,A,mu,sigma: gauss(x,A,mu,sigma)
        costfunc = cost.LeastSquares(x, y, y_unc, wrapper)

    ### Guess the initial values of the parameters if not provided
    if not len(p0):
        p0 = [np.max(y),              # A
                x[np.argmax(y)],      # mu
                (x[-1]-x[0])/10.0]    # sigma
    if offset_prior:
        p0.append(offset_prior)       # c

    ### Do the fit!
    m = Minuit(costfunc, *p0)
    m.errordef = errordef
    m.print_level = print_level
    m.migrad()

    if param_unc:
        m.minos()
        uncs = []
        for param in m.merrors:
            uncs.append([m.merrors[param].lower, m.merrors[param].upper])
    else:
        uncs = [[0.0,0.0]]*len(p0)

    return dict(vals=list(m.values), uncs=uncs)



def fit_histogram_with_gaussian(bins, bin_vals, bin_unc=[], p0=[], \
                                poisson_unc=False, bin_unc_scale=1.0, \
                                errordef=1.0, print_level=0, \
                                offset_prior=0.0, offset_prior_unc=np.inf, \
                                param_unc=False):
    '''
    Takes an input histogram of the default format output by numpy and 
    maplotlib and fits a Gaussian function to the relation between bin values
    and bin centers. Can incorporate user-provided uncertainties, or use 
    Poissonian uncertainties for bin counts. In the absence of either, 
    fitting weights are set to unity

    INPUTS

        bins - 1D array of bin locations. assumed to be the center of
            the bins, but can also be the edges 

        bin_vals - 1D array of the value of the histogram at the locations 
            provided  by the 'bins' argument. Should be the same length
            as 'bins'

        bin_unc - 1D array of uncertainties of the bin values. Should be 
            the same length as 'bins'

        p0 - list or tuple containing initial values of parameters to 
            try. the function will try to find these by default, but if 
            you know them and want to provide them, by all means

        poisson_unc - boolean specifying if we should generate Poisson
            uncertainties for counts in order to weight the fitting

        bin_unc_scale - single scalar. An optional scale factor when 
            using uniform weights. 'migrad()' doesn't like working with 
            really huge or really small numbers, which isn't a problem
            when you have actual uncertainties to weight the various
            terms in the cost function. Here you can simply set the 
            scale of the uncertainty, assumed uniform for all points

        errordef - argument passed to 'Minuit()'. See that documentation.
            Sets the "Delta chi-squared", i.e. how far up the cost function
            you climb from the minimum to determine one standard deviation 
            of the inferred best parameter

        print_level - argument passed to 'Minuit()'. See that documentation.
            Integer from 0-3 with increasing levels of debug messages

        offset_prior - float, optional. if a constant offset is known, this
            parameter can constrain the fit to include it

        param_unc - boolean, specifies whether to use MINOS to compute
            uncertainties on the best-fit parameters

    OUTPUT

        popt = [A, mu, sigma] - array of best fit parameter values

    '''

    ### Coerce to numpy arrays, just in case
    bins = np.array(bins)
    bin_vals = np.array(bin_vals)
    if len(bin_unc):
        bin_unc = np.array(bin_unc)

    if len(bins) != len(bin_vals):
        assert len(bins) == len(bin_vals) + 1, \
            "'bins' and 'bin_vals' should be the same length, " \
            + "or 'bins' should be longer by 1 if it indicates " \
            + "the edges of the bin."
        bin_centers = \
            np.mean( np.stack((bins[:-1], bins[1:]), axis=0), axis=0)

    if len(bin_unc):
        assert len(bin_unc) == len(bin_vals), \
            "'bin_vals' and 'bin_unc' should be the same length"

    if len(p0):
        assert len(p0) == 3, "'p0' initial values should be length-3"

    result = fit_gaussian(bin_centers, bin_vals, bin_unc, \
                          poisson_unc=poisson_unc, errordef=errordef, \
                          print_level=print_level, offset_prior=offset_prior, \
                          offset_prior_unc=offset_prior_unc, \
                          param_unc=param_unc)

    return result





def generate_histogram_and_fit_gaussian(\
        data_vector, bins=20, range=None, density=True, poisson_unc=False, \
        bun_unc_scale=1.0, plot=False, errordef=1.0, print_level=0):
    '''
    Takes an input data_vector, computes a histogram then fits a gaussian
    to the computed histogram. Returns the histogram and the fit values.

    This kind of operation is useful for finding the baseline of a digitized
    pulse, or estimating noise from signal/background free datasets

    INPUTS

        data_vector - N-dimensional array of data to be analyzed. the array
            will be automatically flattened to a single axis

        bins - int or sequence argument passed to numpy.histogram()

        range - tuple argument passed to numpy.histogram()

        density - boolean specifying whether to scale the histogram
            bins such that the distribution represents a probability
            distribution function

        poisson_unc - boolean specifying whether to use Poissonian 
            uncertainties when fitting the Gaussian

        bin_unc_scale - single scalar, passed to the function
            'fit_histogram_with_gaussian()' defined in this module

        plot - boolean for plotting the results

        errordef - argument passed to 'Minuit()'. See that documentation.
            Sets the "Delta chi-squared", i.e. how far up the cost function
            you climb from the minimum to determine one standard deviation 
            of the inferred best parameter

        print_level - argument passed to 'Minuit()'. See that documentation.
            Integer from 0-3 with increasing levels of debug messages

    OUTPUTS

        bins - bin centers
    '''

    bin_vals, bin_edges = np.histogram(data_vector.flatten(), bins=bins, \
                                       range=range, density=density)

    results = fit_histogram_with_gaussian( \
                bin_edges, bin_vals, bin_unc=[], p0=[], poisson_unc=False, \
                bin_unc_scale=1.0, errordef=errordef, print_level=print_level)
    popt = results['vals']

    if plot:
        plot_x = np.linspace(bin_edges[0], bin_edges[-1], 200)
        plot_y = gauss(plot_x, *popt)
        label = '$\\mu={{{:0.3g}}}$, $\\sigma={{{:0.3g}}}$'.format(popt[1], popt[2])

        fig, ax = plt.subplots(1,1)
        ax.bar(bin_edges[:-1], bin_vals, width=np.diff(bin_edges), align='edge')
        ax.plot(plot_x, plot_y, lw=3, color='r', label=label)
        ax.legend(loc=0)
        fig.tight_layout()
        plt.show()

    return popt







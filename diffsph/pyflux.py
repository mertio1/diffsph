import pandas as pd
import numpy as np

import csv
import hashlib

from diffsph.profiles.massmodels import *
from diffsph.spectra.synchrotron import *

from diffsph.utils.dictionaries import *
from diffsph.utils.tools import *
from diffsph.utils.consts import *

cache = load_data('pscache')
cache_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../pscache'))

# ######################### ################# ######################### 
# ######################### Exact Predictions ######################### 
# ######################### ################# ######################### 

############################## Emissivity

def synch_emissivity(r, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Model-specific emissivity from synchrotron radiation
    
    :param r: galactocentric distance in kpc
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0` (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)


    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Emissivity in erg/cm :math:`{}^3` /Hz/s/sr
    :rtype: float
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    if not os.path.exists("pscache"):
        os.mkdir("pscache")

    rkeys = kwargs.keys() - ( kwargs.keys() & {'self_conjugate', 'sv', 'width', 'rate'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}   
    
    arg_str = str([nu, var_to_str(galaxy), var_to_str(rad_temp), var_to_str(hyp), ratio, D0, var_to_str(delta), B, manual, high_res, accuracy, sort_kwargs(**rkwargs)]) 
    filehash = hashlib.sha256(arg_str.encode()).hexdigest()

    sc = kwargs.get('self_conjugate', True)
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    bare = 0.    
    
    if 0 <= r < rh: 
        if filehash + '.csv' not in os.listdir(cache_path):
            m = which_N(nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, high_res, accuracy, **rkwargs)
            coeffs = pd.read_csv(
                os.path.join(cache_path, filehash + '.csv' ), header = None
            ).values.flatten()
            
        elif filehash not in cache.keys():
            coeffs = pd.read_csv(
                os.path.join(cache_path, filehash + '.csv' ), header = None
            ).values.flatten()

        else:
            coeffs = cache[filehash]

        bare = cm_to_kpc * sum(
            coeffs[n - 1] * np.sin( n * np.pi * r / rh ) / r for n in range(1, len(coeffs)) 
        )                    

        if hyp in all_names['decay']:
            return 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * bare
        elif hyp in all_names['wimp']:
            return 1e-9 * kwargs['sv']  / 2 ** (not sc) / 2 / kwargs['mchi']  ** 2 / 4 / np.pi * bare
        else:
            return cm_to_kpc**2 * kwargs['rate'] * bare

    return 0.

############################## Brightness

def synch_brightness(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Model-specific brightness from synchrotron radiation
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)


    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: Brightness in Jy/sr
    :rtype: float
    '''    
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    if not os.path.exists("pscache"):
        os.mkdir("pscache")

    rkeys = kwargs.keys() - ( kwargs.keys() & {'self_conjugate', 'sv', 'width', 'rate'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}   
    
    arg_str = str([nu, var_to_str(galaxy), var_to_str(rad_temp), var_to_str(hyp), ratio, D0, var_to_str(delta), B, manual, high_res, accuracy, sort_kwargs(**rkwargs)]) 
    filehash = hashlib.sha256(arg_str.encode()).hexdigest()
    
    sc = kwargs.get('self_conjugate', True)
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    thmax = np.arcsin(rh / dist)
    bare = 0.

    if 0 <= theta < thmax * rad_to_arcmin:
        if filehash + '.csv' not in os.listdir(cache_path):
            m = which_N(nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, high_res, accuracy, **rkwargs)
            coeffs = pd.read_csv(
                os.path.join(cache_path, filehash + '.csv' ), header = None
            ).values.flatten()
            
        elif filehash not in cache.keys():
            coeffs = pd.read_csv(
                os.path.join(cache_path, filehash + '.csv' ), header = None
            ).values.flatten()

        else:
            coeffs = cache[filehash]
            
            
        bare = sum(
            coeffs[n - 1] * f( n, theta / rad_to_arcmin / thmax) for n in range(1, len(coeffs)) 
        )
        
        if hyp in all_names['decay']:
            return 1e23 * 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * bare
        elif hyp in all_names['wimp']:
            return 1e23 * 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * bare
        else:
            return 1e23 * cm_to_kpc**2 * kwargs['rate'] * bare

    return 0.

def synch_TB(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Model-specific brightness temperature from synchrotron radiation
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)


    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Brightness temperature in mK
    '''
    return TB(synch_brightness, theta, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, high_res, accuracy, **kwargs)


############################## Flux density

def synch_flux_density(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Model-specific flux density from synchrotron radiation
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)


    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Flux density in µJy
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    if not os.path.exists("pscache"):
        os.mkdir("pscache")
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'self_conjugate', 'sv', 'width', 'rate'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}   
    
    arg_str = str([nu, var_to_str(galaxy), var_to_str(rad_temp), var_to_str(hyp), ratio, D0, var_to_str(delta), B, manual, high_res, accuracy, sort_kwargs(**rkwargs)]) 
    filehash = hashlib.sha256(arg_str.encode()).hexdigest()
    
    sc = kwargs.get('self_conjugate', True)
    
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    thmax = np.arcsin(rh / dist)
    
    bare = 0.
    
    if filehash + '.csv' not in os.listdir(cache_path):
        m = which_N(nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, high_res, accuracy, **rkwargs)
        coeffs = pd.read_csv(
            os.path.join(cache_path, filehash + '.csv' ), header = None
        ).values.flatten()
            
    elif filehash not in cache.keys():
        coeffs = pd.read_csv(
            os.path.join(cache_path, filehash + '.csv' ), header = None
        ).values.flatten()

    else:
        coeffs = cache[filehash]
        
    bare = sum(
        coeffs[n - 1] * halo_fd( n , theta / rad_to_arcmin, dist, rh) for n in range(1, len(coeffs)) 
    )
                
    if hyp in all_names['decay']:
        return 1e23 * 1e6 * 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * bare
    elif hyp in all_names['wimp']:
        return 1e23 *  1e6 * 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * bare
    else:
        return 1e23 * 1e6 * cm_to_kpc**2 * kwargs['rate'] * bare


# ######################### ####################### ######################### 
# ######################### Approximate Predictions ######################### 
# ######################### ####################### ######################### 

############################## Emissivity


def synch_emissivity_approx(r, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, regime = 'B', manual = False, **kwargs):
    '''
    Model-specific emissivity from synchrotron radiation in the Regime "A", "B" or "C" approximations
    
    :param r: galactocentric distance in kpc
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Emissivity in erg/cm :math:`{}^3` /Hz/s/sr
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
        
    sc = kwargs.get('self_conjugate', True)
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    
    tau0 = rh ** 2 / D0 / cm_to_kpc **2 
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'self_conjugate', 'sv', 'width', 'rate', 'Gamma'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}
    
    if 0 <= r < rh: 
        
        if hyp == 'wimp': 
            return 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * Hem(
                r, galaxy, rad_temp, 'wimp', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'wimp', mchi = kwargs['mchi'], channel = kwargs['channel'])
        elif hyp == 'decay': 
            return 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * Hem(
                r, galaxy, rad_temp, 'decay', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'decay', mchi = kwargs['mchi'], channel = kwargs['channel'])
        else:
            return 1e-9 * cm_to_kpc**3 * kwargs['rate'] *  Hem(
                r, galaxy, rad_temp, 'generic', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'generic', Gamma = kwargs['Gamma'])
        
    return 0.

############################## Brightness

def synch_brightness_approx(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, regime = 'B', manual = False, **kwargs):
    '''
    Model-specific brightness from synchrotron radiation in the Regime "A", "B" or "C" approximations
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Brightness in Jy/sr
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    sc = kwargs.get('self_conjugate', True)
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    tau0 = rh ** 2 / D0 / cm_to_kpc **2 

    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'self_conjugate', 'sv', 'width', 'rate', 'Gamma'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}
    
    if 0 <= theta < np.arcsin(rh / dist) * rad_to_arcmin: 
        
        if hyp == 'wimp': 
            return 1e23 * 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * Hbr(
                theta, galaxy, rad_temp, 'wimp', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'wimp', mchi = kwargs['mchi'], channel = kwargs['channel'])
        elif hyp == 'decay': 
            return 1e23 * 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * Hbr(
                theta, galaxy, rad_temp, 'decay', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'decay', mchi = kwargs['mchi'], channel = kwargs['channel'])
        else:
            return 1e23 * 1e-9 * cm_to_kpc**2 * kwargs['rate'] *  Hbr(
                theta, galaxy, rad_temp, 'generic', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'generic', Gamma = kwargs['Gamma'])

    return 0.
                            
def synch_TB_approx(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, regime = 'B', manual = False, **kwargs):
    '''
    Model-specific brightness temperature in the Regime "A", "B" or "C" approximations
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Brightness temperature in mK
    '''
    return TB(synch_brightness_approx, theta, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, regime, manual, **kwargs)
    
    
def synch_flux_density_approx(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, regime = 'B', manual = False, **kwargs):
    '''
    Model-specific flux density from synchrotron radiation in the Regime "A", "B" or "C" approximations
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Flux density in µJy
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    sc = kwargs.get('self_conjugate', True)
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    tau0 = rh ** 2 / D0 / cm_to_kpc **2 

    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'self_conjugate', 'sv', 'width', 'rate', 'Gamma'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}
    
    if 0 <= theta < np.arcsin(rh / dist) * rad_to_arcmin: 
        
        if hyp == 'wimp': 
            return 1e23 * 1e6 * 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * Hfd(
                theta, galaxy, rad_temp, 'wimp', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'wimp', mchi = kwargs['mchi'], channel = kwargs['channel'])
        elif hyp == 'decay': 
            return 1e23 * 1e6 * 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * Hfd(
                theta, galaxy, rad_temp, 'decay', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'wimp', mchi = kwargs['mchi'], channel = kwargs['channel'])
        else:
            return 1e23 * 1e6 * 1e-9 * cm_to_kpc**2 * kwargs['rate'] *  Hfd(
                theta, galaxy, rad_temp, 'generic', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'generic', Gamma = kwargs['Gamma'])

    return 0.


# ####################################################### Auxiliary functions

def coeff(n, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual = False, **kwargs):
    '''
    n-th coefficient participating in the Fourier-expanded Green's function solution of the CRE transport equation 
        
    .. math:: 
        s_n = h_n\\times X_n
    
    :param n: order of the halo/bulge factor
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: `n`-th coefficient in the `which_N` function  
    '''
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    tau0 = rh ** 2 / D0 / cm_to_kpc **2 
    if not manual:
        ref = kwargs['ref']
        
        if hyp in ('wimp', 'decay'): 
            return h(n, galaxy, rad_temp, hyp, ratio, ref = ref) * X(nu, tau0 / n ** 2 , delta, B, hyp, 
                                                                     mchi = kwargs['mchi'], 
                                                                     channel = kwargs['channel'])
        
        return h(n, galaxy, rad_temp, hyp, ratio, ref = ref) * X(nu, tau0 / n ** 2 , delta, B, 'generic', 
                                                                     Gamma = kwargs['Gamma'])
    
    hkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'Gamma'} )
    hargs = {hk: kwargs[hk] for hk in hkeys}                 
    
    if hyp in ('wimp', 'decay'): 
        return h(n, galaxy, rad_temp, hyp, ratio, manual = True, **hargs) * X(nu, tau0 / n ** 2 , delta, B, hyp, 
                                                                     mchi = kwargs['mchi'], 
                                                                     channel = kwargs['channel'])
        
    return h(n, galaxy, rad_temp, hyp, ratio, manual = True, **hargs) * X(nu, tau0 / n ** 2 , delta, B, 'generic', 
                                                                     Gamma = kwargs['Gamma'])


def which_N(nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Determines at which order should the Fourier-expanded Green's function solution be truncated and stores the associated :math:`s_n = h_n\\times X_n` coefficients as an array in the ``/cache`` folder
        
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
            
    :return: series truncation order *N*
    '''
    arg_str = str([nu, var_to_str(galaxy), var_to_str(rad_temp), var_to_str(hyp), ratio, D0, var_to_str(delta), B, manual, high_res, accuracy, sort_kwargs(**kwargs)])
    
    filename = hashlib.sha256(arg_str.encode()).hexdigest() + ".csv"

    # Create the data directory if it doesn't exist
    if not os.path.exists("pscache"):
        os.mkdir("pscache")
    
    # Initiate loop          
    m = 1

    # Inititate total sums
    
    S = coeff(1, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, **kwargs)
    Slow = S
    
    # Inititate series 
    slist = [S, coeff(2, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, **kwargs)]
    
    if high_res:

        # Loop
        while np.abs(slist[-1] / S) > 0.01 * accuracy:
            m += 1
            S += slist[-1]

            # Add a new entry to list if the accuracy is < 1% 
            slist.append(coeff(m + 1, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, **kwargs) )

    # Loop
    while np.abs(slist[-1] / Slow / m) > 0.01 * accuracy:    
        m += 1
        S += slist[-1]
        Slow += slist[-1]  * (-1) ** (m - 1) / m 

        # Add a new entry to list if the accuracy is < 1% 
        slist.append(coeff(m + 1, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, **kwargs) )

    # Save file once the desired accuracy is achieved     
    with open(os.path.join("pscache", filename), mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        for row in slist:
            writer.writerow([row])
    return m



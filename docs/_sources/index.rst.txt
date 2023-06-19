.. diffsph documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation page for ``diffsph``
==================================

``diffsph`` is a Python package that computes diffuse signals (e.g. brightness) emitted by Milky-Way satellite dwarf spheroidal galaxies. The underlining physics is mostly based on [#]_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   modules

|

Architecture
============

The code's structure can be summarized as follows: 

.. image:: flowchart.png
   :width: 1000

Besides being able to compute fluxes with the ``pyflux`` (:py:mod:`diffsph.pyflux`) module, the ``limits`` (:py:mod:`diffsph.limits`) module enables users to produce 2\ :math:`\sigma` limits on e.~g. annihilation cross sections or decay rates of dark matter particles.

|

Examples
========

``pyflux`` module
-----------------

In order to get familiar with the code, the user can use the following set of commands in order to generate the figure below::

    from diffsph import pyflux as pf
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    # Angle grid in arcmin 
    
    theta_grid = [15 * i / 1000 for i in range(0,1000)]
    
    # List of satellite galaxies
    
    dsph_list = ['Ursa Major II', 'Fornax', 'Ursa Minor', 'Sextans']
    
    # diffsph's computations at nu = 150 MHz and for the given model
    
    Inu = [[pf.synch_brightness(th, nu = .150, galaxy = gal, rad_temp = 'HDZ', 
                                     hyp = 'wimp', ref = '1408.0002', sv = 3e-26, 
                                     mchi = 10, channel = 'mumu', high_res = True, 
                                     accuracy = .1) 
                 for th in theta_grid] 
                for gal in dsph_list]
    
    # Plots
    
    plt.plot(theta_grid, Inu[0], "k", label = dsph_list[0])
    plt.plot(theta_grid, Inu[1], "--k", label = dsph_list[1])
    plt.plot(theta_grid, Inu[2], ":k", label = dsph_list[2])
    plt.plot(theta_grid, Inu[3], "-.k", label = dsph_list[3])
    plt.legend()
    plt.xlabel('$\\theta$ (arcmin)', size = 'large');
    plt.ylabel('$I_\\nu$ (Jy/sr)', size = 'large');
    plt.title('Brightness profiles with diffsph');
    plt.text(7.5, 630, '$\chi\chi\, \\to\, \mu^+\mu^-$', 
             horizontalalignment = 'center', size = 'large');

.. image:: example_pyflux.png
   :width: 1000
   
In this example, the ordering of the elements in the list ``dshp_list`` is not arbitrary. It was deliberately chosen in such a way that the total flux density of the first element is the largest while the rest are sorted in decreasing order. The following command line allows one to assess this quantitatively::

    # Total flux density for each galaxy in mJy
    Snu = [[gal, 1e-3 * pf.synch_flux_density(30, nu = .150, galaxy = gal, 
                                              rad_temp = 'HDZ', hyp = 'wimp', 
                                              ref = '1408.0002', sv = 3e-26, 
                                              mchi = 10, channel = 'mumu', 
                                              high_res = True, accuracy = .1)] 
                for gal in dsph_list]
                
    # Print
    print(Snu)
    
    >>> [['Ursa Major II', 5.205501952938485], ['Fornax', 3.2288461400011492], 
    ['Ursa Minor', 2.8903428197065293], ['Sextans', 1.6467270289999836]]


``limits`` module
-----------------

The following example shows how to obtain limits on e.~g. the decay rate of dark matter particles using the given noise level of a non-detection image of Draco. It takes (without parallelization) about one hour to compute all::

    from diffsph import limits as lims
    
    # DM mass grid in GeV 
    
    mass_grid = [100 * 10 ** (3 * i / 1000) for i in range(0,1000)]
    
    # List of decay channels
    
    ch_list = ['WW', 'ZZ', 'hh', 'nunu', 'mumu', 'tautau', 'qq', 'cc', 'bb', 'tt']
    
    # diffsph's computations at nu = 150 MHz and for the given image
    
    rates = [[lims.decay_rate_limest(nu = .15, rms_noise = 100, beam_size = 20, 
                                     galaxy = 'Draco', rad_temp = 'HDZ', mchi = mch, 
                                     channel = ch, high_res = True, accuracy = .1, 
                                     ref = '1408.0002') 
              for mch in mass_grid]
             for ch in ch_list]
    
    # Plots
    
    [plt.loglog(mass_grid, rates_Draco[i], label = ch_list[i], ls = '--') for i in range(0, 3)]
    [plt.loglog(mass_grid, rates_Draco[i], label = ch_list[i], ls = ':') for i in range(3, 6)]
    [plt.loglog(mass_grid, rates_Draco[i], label = ch_list[i]) for i in range(6, len(ch_list))]
    plt.ylim([2e-24,8e-21]);
    plt.xlim([200,1e5]);
    plt.legend(loc = 'upper right', ncols = 5)
    plt.xlabel('$m_\chi$ (GeV)', size = 'large');
    plt.ylabel('$\Gamma_{dec}$ (s${}^{-1}$)', size = 'large');
    plt.title('diffsph 2$\sigma$ limit estimates on DM decay rates');
    plt.text(8e4, 4e-24, 'Draco', horizontalalignment = 'right', size = 'large');

.. image:: example_limits.png
   :width: 1000

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. rubric:: References

.. [#] M. Vollmann, "Universal profiles for radio searches of dark matter in dwarf galaxies", 
   doi:`10.1088/1475-7516/2021/04/068 <https://iopscience.iop.org/article/10.1088/1475-7516/2021/04/068>`_ 
   [arXiv:`2011.11947 [astro-ph.HE] <https://arxiv.org/abs/2011.11947>`_].
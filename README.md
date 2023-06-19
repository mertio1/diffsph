# diffsph
Your Python library to calculate diffuse signals from dwarf spheroidal galaxies!


## Table of Contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Usage](#Usage)
4. [Credits](#Credits)
5. [Contributing](#Contributing)
6. [License](#License)
<!---7. [Future ideas](#Future ideas)--->


## Introduction
The [Python](https://www.python.org/) library *diffsph* offers a fast and interactive way to calculate diffuse emission signals in the radio band 
from dwarf spheroidal galaxies. More specifically, it contains functions for the computation of brightnesses, emissivities and flux densities that are associated with the annihilation or decay of dark matter particles as well as generic astrophysical sources.

<!---  We additionally offer a Jupyter notebook that acts as a tutorial and to perform quick calculations. You can download it [here](https://www.dropbox.com/s/o1ldshiqm31p8cd/tutorial.ipynb?dl=0) (Dropbox download). If you use *diffsph* for your research project, please make sure to cite [this](https://www.google.com) ***Insert correct link!*** paper in your publication! --->


## Installation
The library can be installed either with [pip](https://pypi.org/project/pip/) or with 
 [conda](https://anaconda.org/anaconda/conda) via command line. Just execute one of the lines below, depending on 
the manager you are using.
```bash
pip install diffsph
```
```bash
conda install diffsph
```

### Requirements
*diffsph* requires the following Python libraries to run properly:
* numpy (version 1.18.5 or higher)
* scipy (version 1.5.3 or higher)
* pandas (version 0.24.2 or higher)

## Usage
*diffsph* can be used for a variety of calculations. We will present the most important in this section. 

<!--- For a complete overview, please make sure to take a look at [Vollmann, 2023](https://www.google.com) ***insert correct link!***.

### Computations on diffuse radio signals

### Constraining Dark Matter properties
The library can also be used to constrain the self-annihilation cross-section (eg. as done in [Gajovic, 2023](https://arxiv.org/abs/2303.12155)) 
or the decay rate of WIMPs. The top-level functions for that are coded in [limits.py](limits.py) and can be accessed by using
```
from diffsph import limits
```
Available functions are `calculate_cross_section` and `calculate_decay_rate`. Both functions need galaxy data (dependingof course on the galaxy on which you want to perform your calculation), for example half-light radii, distance and size. You can select the galaxy you need by filling the `ref=` and the `galaxy=` parameters (for the data catalogue and the column, your galaxy is stored in - see [Data handling](#Data handling) for details). Additionally, you have to provide the decay channel (`channel=`) and a turbulence model (`turbulence=`). We include several turbulence models and all decay channels. To avoid missspelling and therefore missleading results, you can use a set of inputs that are translated into the same turbulence model (e.g. `'kol'`, `'kolmogorov'`, `'KOL'`, ... are translated into Kolmogorov's turbulence  model and hence $\delta = 1/3$ ) by a json file. Physical parameter describing the diffusion process are the diffusion-norm parameter `diffusion_norm` (in 10^28 cm^3/s^2), the RMS field strength of the turbulent magnetic field `B_rms` in µG and the  WIMP mass `mDM` in GeV. Please note that the function takes only one mass (int of float) and no list. To plot eg the self-annihilation cross-section vs the WIMP mass, you have to fill your arrays using this function eg like the following:
```
import numpy as np
from diffsph import limits
wimp_masses = np.logspace(0, 4)
cross_sections = [calculate_cross_section(..., mDM=m, ...) for m in wimp_masses]
decay_rates = [calculate_decay_rate(..., mDM=m, ...) for m in wimp_masses]
```
These arrays can then be used for plotting. The final parameters are `observational_freq` and `a_fit`. Both are related to 


### Data handling 
To perform the calculations, data from the galaxy of interest is needed. We store the data in different catalogues 
(related to different papers). Each paper has an own csv file that stores all necessary information, named after the 
arxiv reference. Each file contains several rows, one for each galaxy. To load a specific data catalogue, the function 
`load_data()` is used internally. In top-level functions, the user provides the parameters `ref` (for the cataloge) and 
`galaxy` for the row. The `ref` is used as a parameter in `load_data()` and should contain either the first author's 
surname and the year of publishment (eg `ref = 'Geringer-Sameth, 2015'`) or the arxiv reference (eg `ref = '1408.0002'`). 
To select a galaxy within a catalogue, you should provide the galaxy's name as the parameter (eg `galaxy = 'Canes Venatici I'`).
New data can be added to an existing catalogue by using the `add_dwarf()` function in `tools` and can be used eg with+
```
from diffsph import tools
add_dwarf(...)
```
The `add_dwarf` funciton takes an arxiv reference
as parameter (beside the expected galaxy data list or dictionary) to make sure the right catalogue is selected. If no catalogue with
the reference is found, a new one is created (at the moment, you have to add the new reference to the function `load_data()`
to make sure the new catalogue can be used by top-level functions). To remove a galaxy from a catalogue, you can use the function `remove_dwarf()`.
--->

## Credits
Key contributors are [Dr. Martin Vollmann](https://github.com/mertio1), [Finn Welzmüller](https://github.com/FinnWelzmueller)
and [Lovorka Gajovic](https://github.com/lovork). Make sure to get in touch with us in case you want to contribute as well! 


## Contributing
Missing yourself on the list above? This project is open source, so pull requests and contributions are highly welcome! 
Also providing new galaxy data for the community is gladly seen. In this case, please drop us a message 
(directly via E-Mail or as a GitHub Issue) and we get in touch with you for the details as soon as possible! Thank you for your help!

## License
This project runs under the MIT License.

<!---
## Future ideas
- Find a better way to store galaxy data
- Translate first-author and arxiv formats into each other more easily

--->

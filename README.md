# diffSph
Your *Python* library to calculate diffuse signals from dwarf spheroidal galaxies!


## Table of Contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Usage](#Usage)
4. [Credits](#Credits)
5. [Contributing](#Contributing)
6. [License](#License)


## Introduction
The [Python](https://www.python.org/) library *diffSph* offers a fast and interactive way to calculate diffuse emission signals from dwarf spheroidal galaxies. Annihilation/decay of dark matter particles as well as generic astrophysical sources of cosmic-ray electrons and positrons can be investigated using this tool.

## Installation
Use Git or checkout with SVN using the web [URL](https://github.com/mertio1/diffsph.git), e. g.:
```bash
git clone https://github.com/mertio1/diffsph.git
```

or 
```bash
svn co https://github.com/mertio1/diffsph.git
```

For global installations, while in the *diffSph*’s main folder type:
```bash
python setup.py bdist_wheel
```

and (after a few minutes)
```bash
pip install .
```

## Usage
*diffSph* can be used for a variety of calculations. Particularly interesting are

* Emission profiles for indirect Dark Matter detection with gamma rays (*J*-, *D*- factors)

![jfactor](/figs/promptcolor.png)

* Synchtrotron-radiation emission profiles

![synchmap](/figs/synchcolor.png)

* Upper bounds on Dark Matter annihilation cross sections

![sigmav](/figs/svlimits.png)


### Modules
The cornerstones of *diffSph* are the `spectra`, `profiles` and `utils` modules and the top-level functions can be found in `pyflux` and `limits` as depicted in the figure below

![architecture](/figs/mindmap_diffsph.png)


## Credits
Key contributors are [Martin Vollmann](https://github.com/mertio1), [Finn Welzmüller](https://github.com/FinnWelzmueller)
and [Lovorka Gajović](https://github.com/lovork). Make sure to get in touch with us in case you want to contribute as well! 

If you are using *diffSph* please cite the code paper [Vollmann, Welzmüller, Gajović 2024][paper 1]. If you are using for synchrotron-radiation studies please also refer to [Vollmann 2020][paper 2]

## License
This project runs under the MIT License.

<!--- References --->

[paper 1]: <https://arxiv.org> (M. Vollmann, F. Welzmüller , L. Gajović, "diffSph: a Python tool to compute diffuse signals from dwarf spheroidal galaxies")

[paper 2]: <https://arxiv.org/abs/2011.11947> 'M. Vollmann, "Universal profiles for radio searches of dark matter in dwarf galaxies"'

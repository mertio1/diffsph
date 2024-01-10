from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name = 'diffsph',
      version = '0.0.10',
      description = 'A tool to compute diffuse signals from dwarf spheroidal galaxies',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      url = 'https://github.com/mertio1/diffsph',
      author = 'Martin Vollmann',
      author_email = 'martin.vollmann@gmail.com',
      license = 'MIT',
      install_requires = ['numpy', 'scipy', 'pandas'],
      extras_require = dict(dev = ['pytest', 'twine']),
#       packages = find_packages(where=""),
#       package_dir={"": "diffsph"},
#       package_data={"/profiles/data": ["*.csv"]},
      packages = ['diffsph', 
                  'diffsph.profiles',
                  'diffsph.profiles.data',
                  'diffsph.spectra',
                  'diffsph.spectra.Interpolations',                  
                  'diffsph.spectra.Interpolations.delta',
                  'diffsph.spectra.Interpolations.delta.kol',
                  'diffsph.spectra.Interpolations.delta.kol.synrad',
                  'diffsph.spectra.Interpolations.delta.kol.synrad.koldm',                  
                  'diffsph.spectra.Interpolations.delta.kra',
                  'diffsph.spectra.Interpolations.delta.kra.synrad',
                  'diffsph.spectra.Interpolations.delta.kra.synrad.kradm',                  
                  'diffsph.spectra.Interpolations.delta.sixth',
                  'diffsph.spectra.Interpolations.delta.sixth.synrad',
                  'diffsph.spectra.Interpolations.delta.sixth.synrad.sxdm',                  
                  'diffsph.spectra.Interpolations.delta.twth',
                  'diffsph.spectra.Interpolations.delta.twth.synrad',
                  'diffsph.spectra.Interpolations.delta.twth.synrad.twthdm',                  
                  'diffsph.spectra.Interpolations.input',
                  'diffsph.spectra.Interpolations.input.lm',                  
                  'diffsph.utils'],
      include_package_data=True,
      classifiers = [
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'],
     )
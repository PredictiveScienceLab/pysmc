#!/usr/bin/env python


from numpy.distutils.core import setup


setup(name='PySMC',
      description='Sequential Monte Carlo (SMC) for sampling complicated probability densities.',
      author='Ilias Bilionis',
      author_email='ibilion@purdue.edu',
      url='https://github.com/ebilionis/pysmc',
      download_url='https://github.com/ebilionis/pysmc/tarball/1.0',
      keywords=['sequential monte carlo', 'markov chain monte carlo', 'metropolis-hastings',
                'multimodal probability densities', 'particle methods'],
      version='2.1',
      packages=['pysmc'])

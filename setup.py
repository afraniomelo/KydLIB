#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  17 20:03:43 2022

@author: afranio
"""

import setuptools
import os

with open('README.md') as f:
    README = f.read()
    
requirements = os.path.dirname(os.path.realpath(__file__))+'/requirements.txt'

if os.path.isfile(requirements):
    with open(requirements) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    author="Afrânio Melo",
    author_email="afraeq@gmail.com",
    name='kydlib',
    description='Routines for exploratory data analysis.',
    license="MIT",
    version='0.1.0',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/afraniomelo/kydlib',
    packages=setuptools.find_packages(include=['kydlib','kydlib.*']),
    python_requires=">=3.8",
    install_requires=install_requires
)
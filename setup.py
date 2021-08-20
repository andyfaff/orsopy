#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pyyaml', 'dataclasses_json']

test_requirements = ['pytest>=3', 'jsonschema']

setup(
    author="Andrew R. McCluskey",
    author_email='andrew.mccluskey@ess.eu',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="orsopy is a python implementation of ORSO functionality, such as file format phasing",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='orsopy',
    name='orsopy',
    packages=find_packages(include=['orsopy', 'orsopy.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/reflectivity/orsopy',
    version='0.0.1',
    zip_safe=False,
)

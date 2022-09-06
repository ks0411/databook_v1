from setuptools import find_packages, setup

setup(
    name = 'databook', 
    version='0.1', 
    packages=['databook'], # packages=find_packages(),
    install_requires=['pandas'],
)

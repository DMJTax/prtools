from setuptools import setup, find_packages
import os

setup(
   name='prtools',
   version='1.2.1',
   description='Bare-bones implementation of Prtools for Python',
   author='D.M.J. Tax',
   author_email='',
   packages=['prtools'],
   install_requires=['scikit-learn', 'numpy', 'matplotlib', 'requests'],
   package_data={
       'prtools': [os.path.join('data','*.mat')],
       },
)

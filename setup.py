from setuptools import setup
import os

setup(
   name='prtools',
   version='1.1',
   description='Bare-bones implementation of Prtools for Python',
   author='D.M.J. Tax',
   author_email='',
   packages=['prtools'],
   install_requires=['sklearn', 'numpy', 'matplotlib', 'requests', 'mlxtend'],
   package_data={
      'prtools': [ os.path.join('data', '*.mat') ],
   },
)

from setuptools import setup

 setup(
   name='etpro2vec',
   version='0.1.0',
   author='Ichcha Manipur and Maurizio Giordano',
   author_email='ichcha.manipur@icar.cnr.it',
   packages=['package_name', 'package_name.test'],
   #scripts=['bin/script1','bin/script2'],
   #url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='The netpro2vec graph-embedding method',
   long_description=open('README.txt').read(),
   install_requires=[
       "python-igraph",
       "tqdm",
       "pandas",
       "scipy"
   ],
)

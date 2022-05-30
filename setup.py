from setuptools import setup

 setup(
   name='netpro2vec',
   version='0.1.0',
   author='Maurizio Giordano',
   author_email='maurizio.giordano@icar.cnr.it',
   packages=['package_name', 'package_name.test'],
   #scripts=['bin/script1','bin/script2'],
   #url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='The netpro2vec graph-embedding method',
   long_description=open('README.md').read(),
   install_requires=[
       "tqdm",
       "pandas",
       "scipy"
   ],
)

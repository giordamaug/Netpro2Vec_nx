from setuptools import setup

 setup(
   name='Netpro2vec_nx',
   version='0.1.0',
   author='Maurizio Giordano',
   author_email='maurizio.giordano@icar.cnr.it',
   packages=['netpro2vec'],
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

from setuptools import setup, find_packages


setup(
   name='deep_learning',
   version='0.0.1',
   description='Deep Learning model implementations',
   author='Zurab Dzindzibadze',
   author_email='dzindzibadzezurabi@gmail.com',
   packages=find_packages(
      exclude=['examples', 'examples.*', 'scripts', 'scripts.*', 'test', 'test.*'], 
   ),
)
from setuptools import setup, find_packages


setup(
   name='deep_learning',
   version='0.0.2',
   description='Deep Learning model implementations',
   author='Zurab Dzindzibadze',
   author_email='dzindzibadzezurabi@gmail.com',
   packages=find_packages(
      exclude=['examples', 'scripts'], 
   ),
)
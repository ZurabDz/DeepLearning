"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages


setup(
    name="deep",  # Required
    version="2.0.0",  # Required
    description="Deep Learning implementation",
    url="https://github.com/pypa/sampleproject",
    author="Zurab Dzindzibadze",
    author_email="dzindzibadzezurabi@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="sample, setuptools, development",
    package_dir={"": "."},
    packages=find_packages('.'),
    python_requires=">=3.7",
)

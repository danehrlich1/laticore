from distutils.core import setup
from setuptools import find_packages
setup(
    name='laticore',
    version='0.1.11',
    packages=find_packages(),
    package_data = {},
    include_package_data=True,    # include everything in source control
    license='Copyright Konture, Inc.',
    description="Common Packages for Latitude Systems",
    long_description=open('README.md').read(),
    install_requires=[
        "boto3~=1.7",
        "bson~=0.5",
        "h5py~=2.8",
        "Keras~=2.1",
        "numpy~=1.14",
        "pymongo~=3.6",
        "redis~=2.10",
        "scikit-learn~=0.19",
        "scipy~=1.0",
        "sklearn~=0.0",
        "tensorflow~=1.6",
     ],
     url = "https://github.com/konture/latitude-core",
     author = "Scott Crespo",
     author_email = "scott@konture.io",
     python_requires='~=3.3',
)

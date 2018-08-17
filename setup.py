from distutils.core import setup

setup(
    name='latitcore',
    version='0.0.0',
    packages=['laticore',],
    package_data = {},
    include_package_data=True,    # include everything in source control
    license='Copyright Konture, Inc.',
    description="Common Packages for Latitude Systems",
    long_description=open('README.md').read(),
    install_requires=[
        "boto3>=1.7,<2",
        "bson>=0.5,<1",
        "h5py>=2.8,<3",
        "Keras>=2.1,<3",
        "numpy>=1.14,<2",
        "pymongo>=3.6<4",
        "redis>=2.10,<3",
        "scikit-learn>=0.19,<1",
        "scipy>=1.0,<2",
        "sklearn>=0.0,<1",
        "tensorflow>=1.6,<2",
     ],
     url = "https://github.com/konture/latitude-core",
     author = "Scott Crespo",
     author_email = "scott@konture.io",
     python_requires='~=3.3',
)

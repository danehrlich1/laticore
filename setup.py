from distutils.core import setup

setup(
    name='latitude-core',
    version='0.0.0',
    packages=['metricsets',],
    #package_data = {},
    include_package_data=True,    # include everything in source control
    license='Copyright Konture, Inc.',
    description="Common Packages for Latitude Systems",
    long_description=open('README.md').read(),
    install_requires=[
        "numpy>=1.14,<2",
        "scikit-learn>=0.19,<1",
        "scipy>=1.0,<2",
        "sklearn>=0.0,<1",
     ],
     url = "https://github.com/konture/latitude-core",
     author = "Scott Crespo",
     author_email = "scott@konture.io",
     python_requires='~=3.3',
)

from setuptools import setup, find_packages

setup(
    name='cross_correlation',
    version='0.1',
    # url='https://github.com/mypackage.git',
    author='Jan Rathfelder',
    author_email='jan.rathfelder@bayer.com',
    description='cross-correlation for time-series data',
    packages=find_packages(),    
    install_requires=['scipy', 'pandas==1.2', 'numpy==01.18.1', 'typing', 'numbers'],
)
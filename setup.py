from setuptools import setup, find_packages

setup(
    name='gpu_nn',
    version='0.1.0',
    description='gpu accelrated neural network from cuda and numpy',
    author='Ido Goldberg',
    author_email='ido.goldb@gmal.com',
    url='https://github.com/gogoil/GPU_NN_from_scratch',  # Replace with your project's URL
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'License :: OSI Approved :: MIT License',  # Choose your license
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',  # Add other dependencies as needed
        'numba'
    ],
)

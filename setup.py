from setuptools import setup, find_packages

setup(
    name='noma-ml-resource-allocation',
    version='1.0.0',
    author='b222053-ctrl',
    description='Machine learning-based resource allocation for NOMA wireless networks',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
from setuptools import setup, find_packages

setup(
    name="graph_data_processing",
    version="0.1",
    packages=find_packages(include=["graph_dp"]),
    install_requires=[
        "torch_numpy_process>=0.1",  # Dependency on the 'torch_numpy_process' package
        "numpy>=1.24.3",
        "pykeops==2.1.2",
        "pytest>=7.2.0",
    ],
)

from setuptools import find_packages, setup

setup(
    name="mapAD",
    version="0.1.0",
    description="Anomaly detection benchmarking toolkit for single-cell reference mapping",
    author="Zikang Yin",
    author_email="yzk23@mails.tsinghua.edu.cn",
    packages=find_packages(),
    install_requires=[
        "scanpy",
        "anndata",
        "numpy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "mapqc",
        "jupyter",
    ],
    python_requires=">=3.10",
)

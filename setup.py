from setuptools import setup, find_packages

requirements = [
    "torch",
    "scipy",
    "jupyter",
    "matplotlib",
]

dev_requires = [
    "black",
    "flake8",
    "pytest",
    "coverage",
]

setup(
    name="btgp",
    version="0.1",
    description="BTGP",
    author="Anonymous Authors",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requires},
)

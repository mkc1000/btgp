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
    name="efficient_gp",
    version="0.1",
    description="O(n) GP",
    author="Anonymous Authors",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requires},
)

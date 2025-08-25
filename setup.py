from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    required_packages = f.read().splitlines()

setup(
    name="fast_urgent_eval",
    version="0.1",
    packages=find_packages(),
    install_requires=required_packages,
)

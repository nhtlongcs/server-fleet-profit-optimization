from setuptools import setup, find_packages

setup(
    name='techarena',
    version='0.1',
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        # List any dependencies your CLI may have
    ],
    entry_points={
        'console_scripts': [
            'fleet-eval=baseline.evaluator:main',
        ],
    },
)
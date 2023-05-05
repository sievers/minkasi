from setuptools import Extension, setup
import ctypes
import subprocess
import os

try:
    mylib=ctypes.cdll.LoadLibrary("libminkasi.so")
except OSError:
    os.environ["prefix"] = "minkasi"
    subprocess.check_call(["make", "-e", "libminkasi"])
try:
    mylib=ctypes.cdll.LoadLibrary("libmkfftw.so")
except OSError:
    os.environ["prefix"] = "minkasi"
    subprocess.check_call(["make", "-e", "libmkfftw"])

setup(
    name='minkasi',
    version='1.0.0',
    install_requires=[
        'requests',
        'importlib-metadata',
		'numpy',
		'astropy',
		'scipy'
    ],
    packages=['minkasi'],
    package_data={"minkasi": ["*.so"]},
)

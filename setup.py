import subprocess
import sys, os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Clone and install allegro and nequip package 
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/mir-group/allegro.git"])
os.chdir("allegro")
subprocess.run(["pip", "install", "."])
os.chdir("../")

subprocess.run(["git", "clone", "-b", "masks", "https://github.com/mir-group/nequip.git"])
os.chdir("nequip")
subprocess.run(["pip", "install", "."])
os.chdir("../")

setup(
    name='abinitio_train',
    version='0.1.0',
    author='Gabriele Tocci',
    description='Workflow to train nequip and allegro models and run MD with CP2K',
    packages=find_packages(),
    install_requires=requirements,        
    entry_points={
        'console_scripts': [
            'abinitio-train-workflow=train_workflow.workflow:main',
        ],
    },
)

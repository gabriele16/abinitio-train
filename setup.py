from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='abinitio_train',
    version='0.1.0',
    author='Gabriele Tocci',
    description='Workflow to train nequip and allegro models and run MD with CP2K',
    packages=find_packages(),
    install_requires=requirements[:-1],        
    dependency_links = [requirements[-1]],
    entry_points={
        'console_scripts': [
            'abinitio-train-workflow=train_workflow.workflow:main',
        ],
    },
)

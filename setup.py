"""Setup file for piglot package
"""
# Import modules
# --------------
from setuptools import setup, find_packages
import os

# Get path of the package, where steup.py is located
here = os.path.abspath(os.path.dirname(__file__))

# Read the verison number
with open(os.path.join(here, 'VERSION')) as versionFile:
    version = versionFile.read().strip()

# Store the README.md file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    longDescription = f.read()

setup(
    # Project name
    name='piglot',

    # Version from the version file
    version=version,

    # Short description
    description='Parameter Identification by Global Optimisation',

    # Long descriptionf from README.md
    long_description=longDescription,
    long_description_content_type='text/markdown',

    # Github url
    url='https://github.com/CM2S/piglot',

    # Authors
    author='Ana Francisca Alves, Rui Coelho @CM2S, FEUP',
    author_email='afalves@fe.up.pt, ruicoelho@fe.up.pt',

    # Licensing
    licence='MIT',

    # Classifiers (selected from https://pypi.org/classifiers/)
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        # Python version obtained with https://pypi.org/project/check-python-versions/
        'Operating System :: POSIX :: Unix',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'
    ],

    # Keywords
    keywords='python optimisation parameter-identification constitutive-model',

    # Project URLs
    project_urls={
        # 'Documentation': 'https://packaging.python.org/tutorials/distributing-packages/',
        'Source': 'https://github.com/CM2S/piglot',
        'Tracker': 'https://github.com/CM2S/piglot/issues',
    },

    # Python version compatibility
    python_requires='>=3.9',

    # Packages provided
    packages=find_packages(),

    # Scripts provided
    entry_points={
        'console_scripts': [
            'piglot = piglot.bin.piglot:main',
            'piglot-plot = piglot.bin.piglot_plot:main',
        ]
    },

    install_requires=[
        'numpy',
        'tqdm',
        'pandas',
        'sympy',
        'scipy>=1.7',
        'torch',
        'botorch',
        'Pillow',
        'matplotlib',
        'PyYAML'
    ],

    extras_require={'lipo': ['lipo'],
                    'bayesian': ['bayes_opt'],
                    'bayes_sk': ['bask'],
                    'genetic': ['geneticalgorithm'],
                    'pso': ['pyswarms'],
                    'full': ['lipo', 'bask', 'geneticalgorithm', 'pyswarms', 'botorch']},
    )

import os

import pkg_resources
from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'EffortlessML, Simplify your Machine Learning analysis'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

# Setting up
setup(
        name="EffortlessML", 
        version=VERSION,
        author="Z Su",
        author_email="suzhao10086@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(exclude=["test*"]),
        python_requires=">=3.10",
        install_requires=[
            str(r)
            for r in pkg_resources.parse_requirements(
                open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
            )
        ],
        extra_require={'dev': ['pytest']},
        
        keywords=['python', 'machine learning'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
)
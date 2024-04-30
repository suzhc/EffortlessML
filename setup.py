from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'EffortlessML'
LONG_DESCRIPTION = 'EffortlessML'

# Setting up
setup(
        name="EffortlessML", 
        version=VERSION,
        author="Z Su",
        author_email="suzhao10086@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['scikit-learn', 'xgboost'],
        
        keywords=['python', 'machine learning'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
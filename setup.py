from setuptools import setup

setup(
    name='nestedcvtraining',
    version='1.1',
    packages=['nestedcvtraining', 'nestedcvtraining.utils'],
    install_requires=['pandas', 'imblearn', 'numpy', 'scikit-learn', 'scikit-optimize'],
    url='https://github.com/JaimeArboleda/nestedcvtraining',
    license='MIT License',
    author='JaimeArboleda',
    author_email='jaime.arboleda.castilla@gmail.com',
    description='Perform Nested Cross Validation for model training'
)

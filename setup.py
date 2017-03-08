from setuptools import setup

SHORT = 'Feng is a module for feature engineering.'
LONG = SHORT + ' For more info check out the README at \'github.com/mewwts/feng\'.'

setup(
    name='feng',
    version='0.2.7',
    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'scikit-learn>=0.18.0'
    ],
    packages=[
        'feng',
        'feng.importance',
        'feng.pipeline',
        'feng.preprocessing'
    ],
    url='https://github.com/mewwts/feng',
    author='Alexander Svanevik, Anders Aagard, Mats Julian Olsen',
    author_email='mats@plysjbyen.net',
    classifiers=(
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
    ),
    description=SHORT,
    long_description=LONG,
    test_suite='test_feng'
)

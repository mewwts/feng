try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import feng

SHORT='Feng is a module for feature engineering'
LONG= SHORT + '. For more info check out the README at \'github.com/mewwts/feng\'.'

setup(
    name='feng',
    version=feng.__version__,
    packages=['feng',
              'feng.importance',
              'feng.pipeline',
              'feng.preprocessing'],
    url='https://github.com/mewwts/feng',
    author=feng.__author__,
    author_email='mats@plysjbyen.net',
    classifiers=(
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
    ),
    description=SHORT,
    long_description=LONG,
    test_suite='test_feng'
)

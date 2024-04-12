import os, shutil
from setuptools import setup, find_packages

setup(
    name='pycuke',
    version='0.0.1',
    description='A source-to-source compiler for automatic code parallelization and optimization',
    packages=find_packages(),
    package_data={'':['*']},
    include_package_data=True,
    editable=True,
    exclude=['build', 'dist', '*.egg-info'],
)

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
paths = [os.path.join(CUR_PATH, 'build'), os.path.join(CUR_PATH, 'pycuke.egg-info'), os.path.join(CUR_PATH, 'dist')]
for i in paths:
    if os.path.isdir(i):
        print('INFO del dir ', i) 
        shutil.rmtree(i)
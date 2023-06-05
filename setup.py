from os import path

import setuptools
from setuptools import setup

extras = {
    'test': ['pytest', 'pytest_cases']
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='sokoban_bazaar',
      version='0.0.1',
      description='A bazaar of sokoban datasets and solver',
      long_description_content_type='text/markdown',
      long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8').read(),
      url='https://github.com/koulanurag/sokoban-bazaar',
      author='Anurag Koul',
      author_email='koulanurag@gmail.com',
      license='MIT License',
      packages=setuptools.find_packages(),
      install_requires=[
      ],
      extras_require=extras,
      tests_require=extras['test'],
      python_requires='>=3.6, <3.12',
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      )
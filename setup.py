from os import path

import setuptools
from setuptools import setup

extras = {
    "test": ["pytest", "pytest_cases", "pytest-cov", "xdoctest"],
    "dev": ["pandas", "py-markdown-table", "wandb", "black"],
    "docs": [
        "Sphinx<6.0,>=4.0",
        "furo",
        "sphinxcontrib-katex",
        "sphinx-copybutton",
        "sphinx_design",
        "myst-parser",
        "sphinx-autobuild",
        "sphinxext-opengraph",
        "sphinx-prompt",
        "sphinx-favicon",
        "nbsphinx>=0.9.3",
        "pandoc>=2.3",
        "myst-nb",
    ],
}

# Meta dependency groups.
extras["all"] = [item for group in extras.values() for item in group]

setup(
    name="sokoban_bazaar",
    version="0.0.1",
    description="A bazaar of sokoban datasets and solver",
    long_description_content_type="text/markdown",
    long_description=open(
        path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8"
    ).read(),
    url="https://github.com/koulanurag/sokoban-bazaar",
    author="Anurag Koul",
    author_email="koulanurag@gmail.com",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=[
        "gym==0.21.0",
        "pyperplan",
        "torch>=2.0.0",
        "gym-sokoban @ git+https://github.com/koulanurag/gym-sokoban@default#egg=gym-sokoban",
    ],
    extras_require=extras,
    tests_require=extras["test"],
    python_requires=">=3.6, <3.12",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

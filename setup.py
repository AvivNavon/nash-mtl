from io import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# get the long description from the README.md file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# get reqs
def requirements():
    list_requirements = []
    with open("requirements.txt") as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements


setup(
    name="nashmtl",
    version="1.0.0",  # Required
    description="Nash-MTL",  # Optional
    long_description="",  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="",  # Optional
    author="",  # Optional
    author_email="",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    # packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    python_requires=">=3.6",
    install_requires=requirements(),  # Optional
)

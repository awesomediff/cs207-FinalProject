import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="awesomediff", # Replace with your own username
    version="1.1.0",
    author="Gabriel Pestre, Claire Yang, Erin Yang, Wanxi Yang",
    author_email="claireyang@g.harvard.edu",
    description="A package for automatic differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awesomediff/cs207-FinalProject",
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',


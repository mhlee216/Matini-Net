import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mhlee",
    version="0.0.1",
    author_email="leemh216@gmail.com",
    description="Matini-Net",
    url="https://github.com/mhlee216/Matini-Net",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
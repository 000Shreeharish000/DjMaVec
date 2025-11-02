from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="djmavec",
    version="0.1.0",
    author="Shree Harish V",
    author_email="shreeharishv@example.com",
    description="AI-Powered Vector Search for Django with MariaDB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/000Shreeharish000/djmavec",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sentence-transformers",
        "mysql-connector-python",
        "Django>=3.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: Django",
        "Framework :: Django :: 3.2",
        "Framework :: Django :: 4.0",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)

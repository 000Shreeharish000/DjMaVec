from setuptools import setup


setup(
    name="djmavec",
    version="0.1.0",
    description="Django + MariaDB vector search utilities (embeddings, field, and search mixins)",
    author="Sbree Harish V",
    packages=["djmavec"],
    install_requires=[
        "django",
        "numpy",
        "sentence-transformers",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Framework :: Django",
        "Intended Audience :: Django Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

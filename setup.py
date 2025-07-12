from setuptools import setup, find_packages

setup(
    name="multimodal_feature_builder",
    version="0.1.0",
    author="Shivam Bhujbal",
    description="A Python library to build multi-modal feature matrices from text, images, categorical and numeric data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shivambhujbal/multimodal-feature-builder",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "transformers",
        "tqdm",
        "scikit-learn",
        "Pillow"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

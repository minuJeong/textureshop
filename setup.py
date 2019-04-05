from setuptools import setup, find_packages


setup(
    name="textureshop",
    author="minu jeong",
    author_email="minu.hanwool@gmail.com",
    license="None",
    url="http://texture_processor.minujeong.com/",

    packages=find_packages(),
    install_requires=[
        "numpy",
        "imageio",
        "moderngl",
        "pytest",
    ],
    include_package_data=True
)

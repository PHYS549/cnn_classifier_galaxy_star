from setuptools import setup, find_packages

setup(
    name='DataAnalysis_Individual_Project-main',  # Replace with your package name
    version='0.1.0',  # Update as needed
    author='Wen-Hua Wu',  # Replace with your name
    author_email='aw106@rice.edu',  # Replace with your email
    description='A package that utilizes astropy, reproject, and other dependencies',
    url='https://github.com/CrazyAncestor/PhD-Research.git',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        'requests',
        'numpy',
        'astropy',
        'reproject',
        'matplotlib',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
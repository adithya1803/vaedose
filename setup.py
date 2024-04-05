from setuptools import setup, find_packages

pkgs = ['torch=2.2.0',
        'torchvision==0.15.2a0',
        'matplotlib=3.7',
        'numpy==1.24.3',
        'pip==23.3.1',
        'pynrrd==1.0.0',
        'scikit-learn==1.3.0',
        'scipy==1.12.0',
        'tqdm==4.65.0']


setup(name='vaedose',
      version=1.0,
      description='dose distribution encoding',
      python_requires='>3.6',
      author='Adithya',
      install_requires=pkgs,
      packages=find_packages(exclude=['docs','tests']))
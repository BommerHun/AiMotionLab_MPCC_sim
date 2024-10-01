from setuptools import setup, find_packages


install_requires = ['glfw','matplotlib','mujoco','numpy','PyOpenGL','scipy','sympy','ffmpeg-python', 'pyyaml']

setup(name='aimotion_f1tenth_simulator',
      version='1.0.0',
      packages=find_packages(),
      install_requires=install_requires)
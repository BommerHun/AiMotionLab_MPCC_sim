from setuptools import setup, find_packages
from setuptools.command.install import install as _install
import os


class InstallCommand(_install):
    def run(self):
        # Your custom action
        from aimotionlab_virtual.util.util import reorganize_paths
        _install.run(self)
        self.execute(reorganize_paths, (os.path.dirname(os.path.abspath(__file__))),
                     msg="Running post install task to configure MuJoCo paths!")

install_requires = ['glfw','matplotlib','mujoco','numpy','PyOpenGL','scipy','motioncapture==1.0a2','sympy','ffmpeg-python']

setup(name='aimotionlab_virtual',
      version='1.0.0',
      packages=find_packages(),
      install_requires=install_requires)

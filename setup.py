from setuptools import setup

setup(name='carracing_gym',
      version='0.0.1',
      install_requires=[
            'gym==0.21.0',
            'pyglet==1.5.11',
            'gym[box2d]',
            'pyvirtualdisplay'
      ] # And any other dependencies required
)


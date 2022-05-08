from setuptools import setup

setup(name='carracing_gym',
      version='0.0.1',
      install_requires=[
            'gym',
            'gym[box2d]',
            'pyglet'
      ] # And any other dependencies required
)

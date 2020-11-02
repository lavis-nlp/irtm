from distutils.core import setup

setup(
    name='RYN',
    version='0.2',
    packages=['ryn'],
    license='MIT',
    long_description=open('README.md').read(),
    entry_points=dict(console_scripts=['ryn=ryn.cli:main']),
)

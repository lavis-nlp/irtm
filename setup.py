from distutils.core import setup

setup(
    name='Ryn',
    version='0.1',
    packages=['ryn'],
    license='MIT',
    long_description=open('README.md').read(),
    entry_points=dict(console_scripts=['ryn=ryn.cli:main']),
)

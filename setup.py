from distutils.core import setup

setup(
    name="IRTM",
    version="0.4",
    packages=["irtm"],
    license="MIT",
    long_description=open("README.md").read(),
    entry_points=dict(console_scripts=["irtm=irtm.cli:main"]),
)

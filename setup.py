import setuptools

with open("README.md", 'r', encoding="utf-8") as f:
    long_desc = f.read()



__version__ = "0.0.0"
repo = "YOLO-zenml-ops"
author = "IVA"
Script = "zenops"

setuptools.setup(
    name=Script,
    version=__version__,
    author=author,
    long_description=long_desc,
    #url=f"http://github.com/"
    package_dir={"":"scripts"},
    packages=setuptools.find_packages(where="srcipts")
)

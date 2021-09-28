from pathlib import Path

import setuptools

here = Path(__file__).parent

with open(str(here / 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(str(here / 'requirements.txt'), 'r') as f:
    install_reqs = f.read().strip()
    install_reqs = install_reqs.split("\n")

setuptools.setup(
    name='minTE',
    version='0.0.2',
    author='imr-framework',
    author_email='imr.framework2018@gmail.com',
    description='Minimum TE MR sequences using PyPulseq',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/imr-framework/minTE',
    packages=setuptools.find_packages(),
    install_requires=install_reqs,
    license='License :: OSI Approved :: GNU Affero General Public License v3',
    include_package_data=True,

)

from setuptools import find_packages
from distutils.core import setup

packages = find_packages('src', exclude='src')
package_dir = {k: 'src/' + k.replace('.', '/') for k in packages}
#package_data = {'tec/ic/ia/pc1/data': ["*.json"]}
#data_files = [('src/tec/ic/ia/pc1/data','src/tec/ic/ia/pc1/data/forks.json')]

setup(
	name = 'tec',
	version = '2.1',
	description = 'Repositorio del curso de IA',
	url = 'https://...',
	author = 'Fabio Mora - Sergio Moya - Gabriel Venegas',
	author_email = "g...@g....com",
	packages = packages,
	package_dir = package_dir,
	#package_data = package_data
	#data_files = data_files
)

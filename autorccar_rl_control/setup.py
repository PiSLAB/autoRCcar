from setuptools import find_packages, setup

package_name = 'autorccar_rl_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'sb3_contrib'],
    zip_safe=True,
    maintainer='gspark',
    maintainer_email='pks87@nate.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_control = autorccar_rl_control.rl_control:main',
        ],
    },
)

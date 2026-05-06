from setuptools import setup

package_name = 'rl_planner'
import os 
from glob import glob  

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'expl_model'), glob('rl_planner/expl_model/*')),
        (os.path.join('share', package_name, 'nav_model'), glob('rl_planner/nav_model/*')),
    ],
    install_requires=['setuptools', 'sensor_msgs_py', 'PyYAML'],
    zip_safe=True,
    maintainer='Yuhong Cao',
    description='CogniPlan ROS2 planner',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "rl_planner = rl_planner.expl_planner:main",
            "expl_planner = rl_planner.expl_planner:main",
            "nav_planner = rl_planner.nav_planner:main",
        ],
    },
)

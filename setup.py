from setuptools import setup

setup(
    name='QuadrotorEnvironment',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=['quadrotor_environment'],
    url='https://github.com/erniejunior/QuadrotorEnvironment',
    license='GPL',
    author='Maximilian Ernestus',
    author_email='maximilian@ernestus.de',
    description='OpenAI Gym Environment for Quadrotors',
    classifiers=[
        'Development Status :: 4 - Beta',
    ],
    keywords='quadrotor quadcopter reinforcement learning gym',
    python_requires='>=3',
    install_requires=[
        "matplotlib",
        "numpy",
        "numpy-quaternion",
        "sortedcontainers",
        "gym",
    ]
)

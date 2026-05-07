from setuptools import setup, find_packages

setup(
    name="articulate",
    version="0.1.0",
    description="Toolkit for articulated body system",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "opencv-python",
        "pybullet",
        "pygame",
        "pyquaternion",
        "transforms3d",
        "trimesh",
        "open3d",
        "plotly",
        "tqdm",
    ],
    python_requires=">=3.8",
) 
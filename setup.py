from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOR = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOR in requirements:
            requirements.remove(HYPEN_E_DOR)
    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Bhavadeesh",
    author_email="saibhavadeesh@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

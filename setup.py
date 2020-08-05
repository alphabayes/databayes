from setuptools import setup, find_packages
import os
import packaging.version

# Determine version on branch name
active_branch = os.environ.get('CI_COMMIT_REF_NAME')

if not(active_branch is None):
    # CI case
    try:
        version = str(packaging.version.Version(active_branch))

    except packaging.version.InvalidVersion:
        if active_branch in ['devel', 'master']:
            version = active_branch
        else:
            version = "unstable"
else:
    # No CI
    version = "unstable"

# with open('requirements.txt') as f:
#     required_pkg = f.read().splitlines()

setup(name='databayes',
      version=version,
      url='https://gitlab.com/alphabayes/databayes',
      author='Roland Donat',
      author_email='roland.donat@gmail.com, roland.donat@alphabayes.fr',
      maintainer='Roland Donat',
      maintainer_email='roland.donat@gmail.com',
      keywords='pronostic datascience machine-learning ',
      classifiers=['Topic :: datascience', 'Topic :: machine-learning '],
      packages=find_packages(
          exclude=[
              "*.tests",
              "*.tests.*",
              "tests.*",
              "tests",
              "log",
              "log.*",
              "*.log",
              "*.log.*"
          ]
      ),
      description='Model as a service machine learning library',
      license='GPL V3',
      platforms='ALL',
      python_requires='>=3.6',
      install_requires=[
          "typing-extensions>=3.7.4.1",
          "pandas>=0.25.0",
          "numpy>=1.16.0",
          "PyYAML>=3.13",
          "pydantic>=1.3",
          "tqdm>=4.45.0",
          "scikit-learn>=0.21.2",
          "pyAgrum==0.17.1",
          "termcolor==1.1.0",
          "dash",
          "dash-bootstrap-components",
      ],
      zip_safe=False,
      )

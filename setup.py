from setuptools import setup, find_packages
import os
# import pkg_resources
# installed_pkg = {pkg.key for pkg in pkg_resources.working_set}

# # Determine version on branch name
# active_branch = os.environ.get('CI_COMMIT_REF_NAME')
# if active_branch is None:
#     if 'git' in installed_pkg:
#         import git
#         local_repo = git.Repo('./')
#         active_branch = local_repo.active_branch.name

# if not(active_branch is None) and ('packaging' in installed_pkg):
#     import packaging.version  # noqa: 401

#     # CI case
#     try:
#         version = str(packaging.version.Version(active_branch))

#     except packaging.version.InvalidVersion:
#         if active_branch in ['devel', 'master']:
#             version = active_branch
#         else:
#             version = "unstable"
# else:
#     # No CI
#     version = None

version = "0.0.23"

setup(name='databayes',
      version=version,
      url='https://github.com/alphabayes/databayes',
      author='Roland Donat',
      author_email='roland.donat@gmail.com, roland.donat@edgemind.fr, roland.donat@alphabayes.fr',
      maintainer='Roland Donat',
      maintainer_email='roland.donat@gmail.com',
      keywords='pronostic datascience machine-learning ',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
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
          "typing-extensions",
          "pandas",
          "PyYAML",
          "pydantic",
          "tqdm",
          "scikit-learn",
          "pyAgrum",
          "colored",
          "gspread_pandas",
          "openpyxl",
          "XlsxWriter",
          # "psycopg2", # not needed for now
          "intervals",  # Really needed ?
          "deepmerge",
      ],
      zip_safe=False,
      )

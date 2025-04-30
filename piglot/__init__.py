"""Main piglot module."""

__title__ = 'piglot'
__author__ = 'CM2S'
__copyright__ = '2025, CM2S'
__license__ = 'MIT'
__version__ = '0.5.1'


# Try to find the piglot version from the git tag
import os
import subprocess
try:
    result = subprocess.run(
        ['git', 'describe', '--tags'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(__file__),
        text=True,
        check=True
    )
    __new_version__ = result.stdout.strip()

    # The Git tag format is expected to start with 'v' followed by the version number
    # (e.g., 'v0.5.1'). This ensures that the tag corresponds to the current version defined
    # in __version__.
    if __new_version__.startswith(f'v{__version__}'):
        __version__ = __new_version__[1:]
except (FileNotFoundError, subprocess.CalledProcessError):
    pass

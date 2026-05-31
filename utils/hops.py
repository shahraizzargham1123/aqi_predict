"""
One place to log into Hopsworks.

The cert_folder bit matters on Windows: the client defaults its certificate
folder to "/tmp", which doesn't exist here, so we point it at the real system
temp dir instead. On Linux (where the GitHub Actions jobs run) this is harmless.
"""

import tempfile

from utils import config


def login():
    import hopsworks
    return hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
        cert_folder=tempfile.gettempdir(),
    )

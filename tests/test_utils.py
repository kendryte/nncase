import os


def in_ci():
    return os.getenv('CI', False)

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.join(HERE, '..', 'scripts'), os.path.join(HERE, '..', 'src')):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure(config):
    config.addinivalue_line('markers', 'smoke: end-to-end runs of the real entry points on tiny settings')
    config.addinivalue_line('markers', 'gpu: needs a CUDA GPU (skipped otherwise)')

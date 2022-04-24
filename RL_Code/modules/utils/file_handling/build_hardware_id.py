import os
import platform


def init_hardware_id():
    path = '/proc/self/cgroup'
    if os.path.exists('/.dockerenv') or os.path.isfile(path) and any('docker' in line for line in open(path)):
        hardware_id = 'docker'
    else:
        hardware_id = platform.node()  # socket.gethostname()
    return hardware_id

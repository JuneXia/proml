import os
import socket
import getpass

SysConfig = {
    "home_path": os.environ['HOMEPATH'],  # Win10_conda: HOMEPATH, Ubuntu: HOME
    "user_name": getpass.getuser(),
    "host_name": socket.gethostname(),
    "proml_path": "dev/proml"
}

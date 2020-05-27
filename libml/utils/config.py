import os
import socket
import getpass

SysConfig = {
    "home_path": os.environ['HOME'],
    "user_name": getpass.getuser(),
    "host_name": socket.gethostname(),
    "proml_path": "dev/proml"
}

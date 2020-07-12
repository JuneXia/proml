import os
import socket
import getpass

SysConfig = {
    "user_name": getpass.getuser(),
    "host_name": socket.gethostname(),
    "proml_path": "dev/proml"
}

if SysConfig['host_name'] in ['SC-202002221016',  # xiaj home win-pc
                              'DESKTOP-G2VDM3M']:
    SysConfig['home_path'] = os.environ['HOMEPATH']
elif SysConfig['host_name'] in ['mt1080', 'ps']:
    SysConfig['home_path'] = os.environ['HOME']
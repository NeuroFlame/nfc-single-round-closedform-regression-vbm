import subprocess
import time
import psutil
import sys


print("Starting the shell script...")
subprocess.Popen(["/bin/bash", "/workspace/runKit/startup/start.sh"])


time.sleep(10)

print("Polling for nvflare process and printing process details for debugging...")
while True:
    process_found = False
    for proc in psutil.process_iter(attrs=['cmdline']):
        cmdline = ' '.join(proc.info['cmdline']
                           ) if proc.info['cmdline'] else ''
        if 'nvflare' in cmdline:
            process_found = True
            print("nvflare process is running...")
            break

    if process_found:
        time.sleep(10)
    else:
        print("nvflare process is not running anymore or not found. Exiting.")
        sys.exit(0)

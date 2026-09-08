"""Run the supported restricted-notebook CLI through a real Linux PTY.

Used under WSL by a hidden Windows deployment watcher. No shell command strings,
stdin forwarding, credentials, or notebook processes are manipulated here.
"""
import errno
import os
import pty
import select
import signal
import sys
import time

pid,master=pty.fork()
if pid==0:
    env=os.environ.copy()
    env.update(INSPIRE_REQUESTS_HTTP_PROXY='http://127.0.0.1:7897',
               INSPIRE_REQUESTS_HTTPS_PROXY='http://127.0.0.1:7897')
    executable='/home/zhangchenxi/.local/bin/inspire'
    os.execvpe(executable,[executable,'--no-env-file',*sys.argv[1:]],env)
deadline=time.monotonic()+150
try:
    while True:
        if time.monotonic()>deadline:
            os.kill(pid,signal.SIGTERM)
            print('Local CLI PTY timed out; inspect idempotent remote dispatch before retrying.',flush=True)
            sys.exit(124)
        ready,_,_=select.select([master],[],[],1.)
        if ready:
            try:
                chunk=os.read(master,65536)
            except OSError as error:
                if error.errno==errno.EIO:
                    break
                raise
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    _,status=os.waitpid(pid,0)
    sys.exit(os.waitstatus_to_exitcode(status))
finally:
    os.close(master)

import sys,contextlib
from inspire.platform.web.browser_api import jupyter_terminal as jt
@contextlib.contextmanager
def existing(session,notebook_id,**kwargs):
    lab=jt._notebook_jupyter_url(session,notebook_id)
    assert lab
    yield jt._JupyterTerminal(lab_url=lab,name='5',ws_url=jt.rtunnel_module._build_terminal_websocket_url(lab,'5'))
jt._jupyter_terminal=existing
from inspire.cli.main import cli
command=sys.argv[1] if len(sys.argv)>1 else 'hostname'
sys.argv=['inspire','notebook','exec','vlm-r11-trust-h200x4-20260907-r4','--workspace','分布式训练空间','--timeout','60',command]
cli()

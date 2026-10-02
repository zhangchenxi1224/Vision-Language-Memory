import time,re,sys,json
from urllib.parse import urlsplit
import requests
original=requests.Session.request
def observed(self,method,url,*args,**kwargs):
    target='ai-notebook-inspire.sii.edu.cn' in urlsplit(url).netloc
    start=time.monotonic()
    if target:
        kind='terminal-api' if '/api/terminals' in urlsplit(url).path else 'jupyter-entrance'
        print('DIAG',method,kind,'begin',flush=True)
        if kind=='terminal-api' and method.upper()=='POST':
            kwargs['json']={}
            print('DIAG testing explicit JSON empty object',flush=True)
            try:
                check=original(self,'GET',url,timeout=(5,10))
                print('DIAG terminal-list status',check.status_code,'content_type',check.headers.get('Content-Type'),flush=True)
                if check.status_code==200:print('DIAG terminal-list names',[x.get('name') for x in check.json()],flush=True)
            except Exception as e:print('DIAG terminal-list exception',type(e).__name__,flush=True)
    try:
        result=original(self,method,url,*args,**kwargs)
    except Exception as e:
        if target:print('DIAG',method,kind,'exception',type(e).__name__,'elapsed',round(time.monotonic()-start,2),flush=True)
        raise
    if target:
        print('DIAG',method,kind,'status',result.status_code,'elapsed',round(time.monotonic()-start,2),'content_type',result.headers.get('Content-Type'),'xsrf',bool(self.cookies.get('_xsrf')),'redirects',[x.status_code for x in result.history],flush=True)
        if 'text/html' in result.headers.get('Content-Type',''):
            title=re.search(r'<title>(.*?)</title>',result.text,re.S|re.I)
            if title:print('DIAG title',re.sub('<[^>]+>','',title.group(1))[:120],flush=True)
            config=re.search(r'<script[^>]+id="jupyter-config-data"[^>]*>(.*?)</script>',result.text,re.S)
            if config:
                data=json.loads(config.group(1))
                from inspire.platform.web.browser_api.rtunnel import _jupyter_server_base
                print('DIAG base-path matches page',urlsplit(_jupyter_server_base(url)).path==data.get('baseUrl'),'terminalsAvailable',data.get('terminalsAvailable'),flush=True)
    return result
requests.Session.request=observed
from inspire.cli.main import cli
sys.argv=['inspire','notebook','exec','vlm-r11-trust-h200x4-20260907-r4','--workspace','分布式训练空间','--timeout','15','hostname']
cli()

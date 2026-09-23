"""Apply unchanged PrefEval four-part judge prompts/parsers/aggregation to readbacks."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'third_party/prefeval_reference'
SYSTEM="You are a helpful assistant in evaluating an AI assistant's reponse. You should be fair and strict and follow the user's instruction"

def load_functions(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    env={'BeautifulSoup':BeautifulSoup,'tqdm':lambda x:x}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env

def invoke(args,prompt):
    if args.provider=='bedrock':
        import boto3
        client=boto3.client('bedrock-runtime',region_name='us-east-1')
        body={'anthropic_version':'bedrock-2023-05-31','max_tokens':100,'system':SYSTEM,
              'messages':[{'role':'user','content':prompt}],'temperature':0.0}
        response=client.invoke_model(modelId=args.model,body=json.dumps(body))
        raw=json.loads(response['body'].read())
        return raw['content'][0]['text'],raw
    key=os.environ[args.key_env]
    body={'model':args.model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
        'max_tokens':100,'temperature':0.0}
    request=urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    for attempt in range(5):
        # Shared account limits may be below the model's advertised quota.
        time.sleep(2)
        try:
            with urllib.request.urlopen(request,timeout=90) as response:
                raw=json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code not in {429,500,502,503,504} or attempt==4:
                raise
            retry_after=error.headers.get('Retry-After') if error.headers else None
            try:
                delay=max(5*2**attempt,float(retry_after or 0))
            except ValueError:
                delay=5*2**attempt
            error.close()
            print(json.dumps({'judge_transport_retry':error.code,'attempt':attempt+1,'wait_seconds':delay}),flush=True)
            time.sleep(delay)
    return raw['choices'][0]['message']['content'],raw

def main(args):
    parse=load_functions(UPSTREAM/'generation_task/llm_based_evaluation_errortypes.py',
        {'parse_explanation_and_answer','parse_preference_and_answer'})
    aggregate=load_functions(UPSTREAM/'generation_task/get_preference_following_accuracy_generation_task.py',{'analyze_errors'})
    args.output.mkdir(parents=True,exist_ok=True)
    for source in args.inputs:
        for line in source.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['task']!='free':
                continue
            uid=hashlib.sha256(line.encode()).hexdigest()
            path=args.output/f'{uid}.json'
            record=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'source_file':str(source),'input':row,
                'judge_model':args.model,'provider':args.provider,'max_tokens':100,
                'model_label':'official_model' if args.model in {'anthropic.claude-3-sonnet-20240229-v1:0','claude-3-sonnet-20240229'} else 'substitute_judge',
                'evaluation_error_analysis':{},'raw_judgments':{},'status':'pending'}
            assert record['judge_model']==args.model and record['provider']==args.provider
            if record['status']=='complete':
                continue
            checks=record['evaluation_error_analysis']
            for metric,filename in [('acknow','check_acknowledge.txt'),('violate','check_violation.txt'),
                                    ('hallucinate','check_hallucination.txt'),('helpful','check_helpful.txt')]:
                if metric in checks:
                    continue
                prompt=(UPSTREAM/'error_type'/filename).read_text(encoding='utf-8')
                if metric=='acknow':
                    prompt=prompt.replace('{end_generation}',row['generated']['raw']).replace('{question}',row['question'])
                elif metric in ['violate','helpful']:
                    prompt=prompt.replace('{preference}',row['preference']).replace('{question}',row['question']).replace('{end_generation}',row['generated']['raw'])
                else:
                    prompt=prompt.replace('{preference}',row['preference']).replace('{assistant_restatement}',checks['acknow']['extract_pref'])
                text,raw=invoke(args,prompt)
                parsed,answer=parse['parse_preference_and_answer' if metric=='acknow' else 'parse_explanation_and_answer'](text)
                record['raw_judgments'][metric]={'prompt':prompt,'response':raw,'text':text}
                # Prevent malformed/missing official XML from being treated as a successful adherence score.
                if ('yes' in answer.lower())==('no' in answer.lower()):
                    record['status']='judge_parse_failure'
                    path.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
                    raise RuntimeError(f'Judge parse failure for {uid}/{metric}; raw response preserved')
                checks[metric]={'answer':answer,'extract_pref' if metric=='acknow' else 'explanation':parsed}
                path.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
            stats,_=aggregate['analyze_errors']([record])
            record['official_aggregation']=stats
            record['correct']=bool(stats['preference_adherence_accuracy'])
            record['status']='complete'
            path.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
            print(json.dumps({'id':uid,'correct':record['correct'],'model_label':record['model_label']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--provider',choices=['bedrock','openai-compatible'],default='bedrock')
    p.add_argument('--model',default='anthropic.claude-3-sonnet-20240229-v1:0')
    p.add_argument('--base-url')
    p.add_argument('--key-env',default='PREFEVAL_JUDGE_API_KEY')
    main(p.parse_args())

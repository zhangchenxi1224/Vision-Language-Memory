"""Run released PrefEval judge prompts/parsers/aggregation on saved RGB answers.

Uses existing AWS/Bedrock or DASHSCOPE_API_KEY configuration; no keys are stored.
The upstream evaluator is imported from its pinned, unmodified checkout.
"""
import argparse
import hashlib
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

REVISION='50795054b5ff5f418d2b768a331d71e480f93331'
MODEL='anthropic.claude-3-sonnet-20240229-v1:0'
SYSTEM="You are a helpful assistant in evaluating an AI assistant's reponse. You should be fair and strict and follow the user's instruction"
FILES=dict(acknow='check_acknowledge.txt',violate='check_violation.txt',
           hallucinate='check_hallucination.txt',helpful='check_helpful.txt')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--glob',default='*/*/*/*.json');p.add_argument('--families',default='T1,T2,T3,O1,O2')
    p.add_argument('--split',choices=['train','dev','all'],default='all');p.add_argument('--profile')
    p.add_argument('--provider',choices=['bedrock','dashscope'],default='bedrock')
    p.add_argument('--model')
    p.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if a.workers<1:raise ValueError('workers must be positive')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.upstream,text=True).strip()!=REVISION:
        raise RuntimeError('Use the registered official PrefEval revision')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.upstream,text=True).strip():
        raise RuntimeError('Official evaluator checkout has tracked modifications')
    sys.path.insert(0,str(a.upstream.resolve()))
    evaluator=importlib.import_module('generation_task.llm_based_evaluation_errortypes')
    scorer=importlib.import_module('generation_task.get_preference_following_accuracy_generation_task')
    model=a.model or ('qwen3.8-max' if a.provider=='dashscope' else MODEL)
    if a.provider=='dashscope':
        from openai import OpenAI
        client=OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],base_url=a.base_url,
                      timeout=60.,max_retries=3)
        endpoint=dict(provider=a.provider,base_url=a.base_url,enable_thinking=False)
        def invoke(prompt):
            response=client.chat.completions.create(model=model,max_tokens=100,temperature=0.,
                messages=[dict(role='system',content=SYSTEM),dict(role='user',content=prompt)],
                extra_body=dict(enable_thinking=False))
            return response.model_dump(),response.choices[0].message.content or ''
    else:
        import boto3
        from botocore.config import Config
        session=boto3.Session(profile_name=a.profile)
        client=session.client('bedrock-runtime',region_name='us-east-1',config=Config(retries={'max_attempts':3,'mode':'standard'}))
        endpoint=dict(provider=a.provider,region='us-east-1')
        def invoke(prompt):
            body=dict(anthropic_version='bedrock-2023-05-31',max_tokens=100,system=SYSTEM,
                      messages=[dict(role='user',content=prompt)],temperature=0.)
            response=client.invoke_model(body=json.dumps(body),modelId=model)
            raw=json.loads(response['body'].read())
            return raw,raw['content'][0]['text']
    templates={metric:(a.upstream/'error_type'/name).read_text(encoding='utf-8') for metric,name in FILES.items()}
    template_hashes={metric:hashlib.sha256(text.encode()).hexdigest() for metric,text in templates.items()}
    identity=dict(upstream_revision=REVISION,model=model,max_tokens=100,temperature=0.,
                  template_sha256=template_hashes,**endpoint)
    # Separate replacement-judge results from the original Bedrock protocol.
    judge_root=a.run/'judges'
    if a.provider!='bedrock' or model!=MODEL:
        judge_root=judge_root/(a.provider+'-'+model.replace('/','_'))
    families=a.families.split(',');all_scored=[];invalid=0
    if not set(families)<=set(('T1','T2','T3','O1','O2')):raise ValueError('Unknown question form')
    def judge_source(source):
        if source.name.startswith('complete-'):return [],0
        data=json.loads(source.read_text(encoding='utf-8'))
        if a.split!='all' and data['split']!=a.split:return [],0
        dest=judge_root/source.relative_to(a.run/'evaluations')
        source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        result=json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else dict(source_sha256=source_sha,
            **identity,rows={})
        if result['source_sha256']!=source_sha:raise RuntimeError('Raw answer changed after judging')
        for key,value in identity.items():
            if result.get(key)!=value:raise RuntimeError('Judge configuration changed: '+key)
        scored=[];malformed=0
        for row in data['rows']:
            if row['task']!='generation' or row['family'] not in families:continue
            family=row['family'];judged=result['rows'].setdefault(family,dict(evaluation_error_analysis={},raw={}))
            checks=judged['evaluation_error_analysis']
            question=row.get('question',row['query'].removesuffix(' (Please respond within 300 words.)'))
            for metric,template in templates.items():
                if metric in checks:continue
                # Same replacement sequence as upstream, including the extracted
                # acknowledgment used by the hallucination question.
                prompt=template
                if metric=='acknow':
                    prompt=prompt.replace('{end_generation}',row['generation']['raw']).replace('{question}',question)
                elif metric in ('violate','helpful'):
                    prompt=prompt.replace('{preference}',data['judge_only_preference']).replace('{question}',question)
                    prompt=prompt.replace('{end_generation}',row['generation']['raw'])
                else:
                    prompt=prompt.replace('{preference}',data['judge_only_preference'])
                    prompt=prompt.replace('{assistant_restatement}',checks['acknow']['extract_pref'])
                raw,text=invoke(prompt)
                if metric=='acknow':
                    preference,answer=evaluator.parse_preference_and_answer(text)
                    checks[metric]=dict(extract_pref=preference,answer=answer)
                else:
                    explanation,answer=evaluator.parse_explanation_and_answer(text)
                    checks[metric]=dict(explanation=explanation,answer=answer)
                judged['raw'][metric]=dict(prompt=prompt,response=raw)
                save(dest,result)
            # Preserve official aggregation even for malformed judge answers,
            # but expose that failure count instead of silently presenting a clean score.
            malformed+=sum(v['answer'].strip().lower() not in ('yes','no') for v in checks.values())
            scored.append(dict(evaluation_error_analysis=checks))
        print(json.dumps(dict(judged=str(source.relative_to(a.run)),families=families)),flush=True)
        return scored,malformed
    sources=sorted((a.run/'evaluations').glob(a.glob))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        try:
            for scored,malformed in pool.map(judge_source,sources):
                all_scored.extend(scored);invalid+=malformed
        except BaseException:
            pool.shutdown(wait=False,cancel_futures=True)
            raise
    stats,count=scorer.analyze_errors(all_scored)
    summary=dict(**identity,glob=a.glob,split=a.split,families=families,
                 count=count,stats=stats,malformed_judge_answers=invalid,
                 protocol='official_prefeval_prompts_parsers_aggregation',
                 original_judge=(a.provider=='bedrock' and model==MODEL))
    # The selection is explicit in this summary; raw judgments retain full paths.
    selection=hashlib.sha256(json.dumps([a.glob,a.split,families]).encode()).hexdigest()[:12]
    save(judge_root/f'summary-{selection}.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

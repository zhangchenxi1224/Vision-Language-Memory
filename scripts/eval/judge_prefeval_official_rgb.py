"""Run released PrefEval judge prompts/parsers/aggregation on saved RGB answers.

Requires an existing AWS/Bedrock configuration; no credentials are stored here.
The upstream evaluator is imported from its pinned, unmodified checkout.
"""
import argparse
import hashlib
import importlib
import json
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
    a=p.parse_args()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.upstream,text=True).strip()!=REVISION:
        raise RuntimeError('Use the registered official PrefEval revision')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.upstream,text=True).strip():
        raise RuntimeError('Official evaluator checkout has tracked modifications')
    sys.path.insert(0,str(a.upstream.resolve()))
    evaluator=importlib.import_module('generation_task.llm_based_evaluation_errortypes')
    scorer=importlib.import_module('generation_task.get_preference_following_accuracy_generation_task')
    import boto3
    from botocore.config import Config
    session=boto3.Session(profile_name=a.profile)
    client=session.client('bedrock-runtime',region_name='us-east-1',config=Config(retries={'max_attempts':3,'mode':'standard'}))
    templates={metric:(a.upstream/'error_type'/name).read_text(encoding='utf-8') for metric,name in FILES.items()}
    families=a.families.split(',');all_scored=[];invalid=0
    if not set(families)<=set(('T1','T2','T3','O1','O2')):raise ValueError('Unknown question form')
    for source in sorted((a.run/'evaluations').glob(a.glob)):
        if source.name.startswith('complete-'):continue
        data=json.loads(source.read_text(encoding='utf-8'))
        if a.split!='all' and data['split']!=a.split:continue
        dest=a.run/'judges'/source.relative_to(a.run/'evaluations')
        source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        result=json.loads(dest.read_text()) if dest.exists() else dict(source_sha256=source_sha,
            upstream_revision=REVISION,model=MODEL,region='us-east-1',max_tokens=100,temperature=0.,rows={})
        if result['source_sha256']!=source_sha:raise RuntimeError('Raw answer changed after judging')
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
                body=dict(anthropic_version='bedrock-2023-05-31',max_tokens=100,system=SYSTEM,
                          messages=[dict(role='user',content=prompt)],temperature=0.)
                response=client.invoke_model(body=json.dumps(body),modelId=MODEL)
                raw=json.loads(response['body'].read());text=raw['content'][0]['text']
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
            invalid+=sum(v['answer'].strip().lower() not in ('yes','no') for v in checks.values())
            all_scored.append(dict(evaluation_error_analysis=checks))
        print(json.dumps(dict(judged=str(source.relative_to(a.run)),families=families)),flush=True)
    stats,count=scorer.analyze_errors(all_scored)
    summary=dict(model=MODEL,upstream_revision=REVISION,glob=a.glob,split=a.split,families=families,
                 count=count,stats=stats,malformed_judge_answers=invalid)
    # The selection is explicit in this summary; raw judgments retain full paths.
    selection=hashlib.sha256(json.dumps([a.glob,a.split,families]).encode()).hexdigest()[:12]
    save(a.run/'judges'/f'summary-{selection}.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

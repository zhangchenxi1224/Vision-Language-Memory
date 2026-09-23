import io
import hashlib
import json
from types import SimpleNamespace
import urllib.error
from unittest.mock import patch

import pytest
from scripts.experiments import prefeval_k1_judge as judge


def args():
    return SimpleNamespace(provider='openai-compatible',key_env='TEST_JUDGE_KEY',
        model='fixed-test-model',base_url='https://example.invalid/v1')


def test_rate_limit_retry_preserves_request_and_server_delay(monkeypatch):
    monkeypatch.setenv('TEST_JUDGE_KEY','test-only-not-a-real-key')
    payload={'choices':[{'message':{'content':'<answer>No</answer>'}}]}
    limit=urllib.error.HTTPError('https://example.invalid',429,'Rate limit',{'Retry-After':'7'},None)
    with patch.object(judge.urllib.request,'urlopen',side_effect=[limit,io.BytesIO(json.dumps(payload).encode())]) as call, \
         patch.object(judge.time,'sleep') as sleep:
        text,raw=judge.invoke(args(),'original prompt')
    assert text=='<answer>No</answer>' and raw==payload
    assert call.call_args_list[0].args[0] is call.call_args_list[1].args[0]
    body=json.loads(call.call_args_list[0].args[0].data)
    assert body['max_tokens']==100 and body['temperature']==0.0
    assert body['messages'][-1]['content']=='original prompt'
    assert [c.args[0] for c in sleep.call_args_list]==[2,7,2]


def test_auth_failure_is_not_retried(monkeypatch):
    monkeypatch.setenv('TEST_JUDGE_KEY','test-only-not-a-real-key')
    denied=urllib.error.HTTPError('https://example.invalid',401,'Unauthorized',{},None)
    with patch.object(judge.urllib.request,'urlopen',side_effect=denied) as call, \
         patch.object(judge.time,'sleep'),pytest.raises(urllib.error.HTTPError):
        judge.invoke(args(),'original prompt')
    assert call.call_count==1


def test_parse_failure_stays_unscored_and_does_not_block_later_rows(tmp_path):
    source=tmp_path/'answers.jsonl'
    rows=[{'task':'free','question':q,'preference':'likes jazz','generated':{'raw':'Jazz fits.'}}
          for q in ['first question','second question']]
    lines=[json.dumps(row) for row in rows]
    source.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    options=args()
    options.inputs=[source]
    options.output=tmp_path/'judgments'
    texts=['<preference>unfinished',
           '<preference>likes jazz</preference><answer>Yes</answer>',
           '<explanation>Fits.</explanation><answer>No</answer>',
           '<explanation>Accurate.</explanation><answer>No</answer>',
           '<explanation>Helpful.</explanation><answer>Yes</answer>']
    with patch.object(judge,'invoke',side_effect=[(text,{'text':text}) for text in texts]) as call:
        judge.main(options)
    assert call.call_count==5
    paths=[options.output/(hashlib.sha256(line.encode()).hexdigest()+'.json') for line in lines]
    failed=json.loads(paths[0].read_text(encoding='utf-8'))
    succeeded=json.loads(paths[1].read_text(encoding='utf-8'))
    assert failed['status']=='judge_parse_failure' and 'correct' not in failed
    assert failed['raw_judgments']['acknow']['text']==texts[0]
    assert succeeded['status']=='complete' and succeeded['correct'] is True
    original=paths[0].read_bytes()
    with patch.object(judge,'invoke',side_effect=AssertionError('Cached outcomes must not be resampled')) as call:
        judge.main(options)
    assert call.call_count==0 and paths[0].read_bytes()==original

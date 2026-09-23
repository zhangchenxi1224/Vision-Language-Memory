import io
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

import copy
import json
from types import SimpleNamespace
import pytest
import torch

from vision_memory.dreamlite.conditioning import encode_native_base_edit_condition, official_mobile_edit_prompt
from scripts.train.train_latent_bank_unet import verify_initialized_baseline_reference, write_json, file_sha256
from test_logical_condition_sampling import matched_baselines
from test_logical_condition_sampling import actual_groups


def test_native_training_uses_the_full_encoder_batch_then_only_conditional_row():
    image=object()
    embeds=torch.randn(3,8,4,requires_grad=True)
    mask=torch.arange(24).reshape(3,8)
    calls=[]
    def encode(**kwargs):
        calls.append(kwargs)
        return embeds,mask
    pipe=SimpleNamespace(encode_prompt=encode)
    result=encode_native_base_edit_condition(pipe,image,'clear the preference',device='cpu',dtype=torch.float32)
    assert calls==[{'mode':'edit','prompts':['','',official_mobile_edit_prompt('clear the preference')],
        'image':image,'device':'cpu','dtype':torch.float32}]
    assert torch.equal(result.prompt_embeds,embeds[2:3]) and torch.equal(result.attention_mask,mask[2:3])
    assert not result.prompt_embeds.requires_grad
    saved=result.prompt_embeds.clone()
    with torch.no_grad():embeds[2].zero_()
    assert torch.equal(result.prompt_embeds,saved)
    pipe.encode_prompt=lambda **kwargs:(embeds[:2],mask[:2])
    with pytest.raises(ValueError,match='three-branch'):
        encode_native_base_edit_condition(pipe,image,'clear',device='cpu',dtype=torch.float32)


def test_control_allows_only_declared_training_condition_change_and_keeps_actual_baseline_gate(tmp_path):
    current,prior,identity,digest=matched_baselines(tmp_path)
    common={**identity,'model_variant':'base','steps':4832,'sampling':{'strategy':'logical_condition'}}
    write_json(current/'identity.json',{**common,'prompt_style':'native_base'})
    write_json(prior/'train/identity.json',{**common,'prompt_style':'official_raw'})
    raw={'condition_sha256':{'condition':'raw-hash'},'scheduler_config':{'steps':1000},
        'additional_protocol_binding':{'train_prompt':'raw event, upstream LoRA example','inference_steps':28}}
    native=copy.deepcopy(raw)
    native['condition_sha256']['condition']='native-hash'
    native['additional_protocol_binding']['train_prompt']='native Base edit, conditional row of upstream three-branch encoding'
    write_json(current/'runtime.json',native)
    write_json(prior/'train/runtime.json',raw)
    with pytest.raises(ValueError,match='prompt_style'):
        verify_initialized_baseline_reference(current,prior,digest)
    assert verify_initialized_baseline_reference(current,prior,digest,native_condition_control=True)['bitwise_trajectories']
    for changed in ({**native,'scheduler_config':{'steps':500}},
                    {**native,'condition_sha256':{'wrong-condition':'native-hash'}},
                    {**native,'additional_protocol_binding':{**native['additional_protocol_binding'],'inference_steps':4}}):
        write_json(current/'runtime.json',changed)
        with pytest.raises(RuntimeError):
            verify_initialized_baseline_reference(current,prior,digest,native_condition_control=True)
    write_json(current/'runtime.json',native)
    phase=current/'baseline'
    payload=torch.load(phase/'sample.pt',weights_only=True)
    payload['latent']=payload['latent']+.01
    torch.save(payload,phase/'sample.pt')
    seal=json.loads((phase/'complete.json').read_bytes())
    seal['artifact_hashes']['sample.pt']=file_sha256(phase/'sample.pt')
    write_json(phase/'complete.json',seal)
    with pytest.raises(RuntimeError,match='not bitwise'):
        verify_initialized_baseline_reference(current,prior,digest,native_condition_control=True)


def test_native_condition_plan_preserves_exact_prior_draws_budget_and_native_inference():
    from scripts.experiments.logical_sampling_protocol import plan as prior_plan
    from scripts.experiments.native_condition_protocol import plan, REFERENCE_RESULT
    bank={'groups':actual_groups()}
    current=plan(bank,'b'*40)
    prior=prior_plan(bank,'a'*40)
    for key in ('sampling','optimizer','optimizer_steps','draws','global_batch','training_seed',
                'inference','transition_validation','prefix_validation','initial_package_manifest_sha256'):
        assert current[key]==prior[key]
    assert current['training_prompt_style']=='native_base'
    assert current['reference_result_sha256']==REFERENCE_RESULT
    assert 'Only training condition encoding changes' in current['budget_change']


def test_collector_selects_only_the_exact_registered_native_training_commit():
    import hashlib
    from scripts.reporting.collect_broader_endpoint import registered_protocol, NATIVE_CONDITION_COMMIT
    from scripts.experiments.native_condition_protocol import plan, REFERENCE_COMMIT
    from scripts.experiments.logical_sampling_protocol import plan as logical_plan
    bank={'groups':actual_groups()}
    commit,registered,digest=registered_protocol(bank,NATIVE_CONDITION_COMMIT)
    assert commit==NATIVE_CONDITION_COMMIT and registered==plan(bank,commit)
    assert digest==hashlib.sha256((json.dumps(registered,indent=2,sort_keys=True)+'\n').encode()).hexdigest()
    prior,registered,_=registered_protocol(bank,REFERENCE_COMMIT)
    assert registered==logical_plan(bank,prior)
    assert registered_protocol(bank,'a'*40)[1]['schema']!=plan(bank,commit)['schema']

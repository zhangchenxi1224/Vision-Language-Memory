import copy
import json
from types import SimpleNamespace
import pytest
import torch

from scripts.train.train_latent_bank_unet import frozen_audit, frozen_versions, trainable_unet_parameters
from vision_memory.training.checkpoint import checkpoint_due, load_training_checkpoint, save_training_checkpoint


def test_reference_baseline_rejects_changed_numeric_outputs_even_with_valid_artifact_hash(tmp_path):
    from scripts.train.train_latent_bank_unet import verify_baseline_reference, file_sha256, write_json
    reference=tmp_path/"reference"
    current=tmp_path/"current/train"
    reference_train=reference/"train"
    write_json(reference_train/"result.json",{"status":"completed"})
    result_sha=file_sha256(reference_train/"result.json")
    write_json(reference/"terminal.json",{"state":"completed","training_result_sha256":result_sha})
    image=torch.ones(1,3,2,2)
    latent=torch.ones(1,4,2,2)
    for root in (reference_train,current):
        write_json(root/"runtime.json",{"condition":"same","sigma":[1.,0.]})
        phase=root/"baseline"
        phase.mkdir()
        torch.save({"image":image,"latent":latent,"noise_seed":7,"trajectory":[latent,latent]},phase/"sample.pt")
        (phase/"generations.jsonl").write_text(json.dumps({"raw":"ambient"})+"\n")
        write_json(phase/"complete.json",{"artifact_hashes":{p.name:file_sha256(p) for p in phase.iterdir()}})
    result=verify_baseline_reference(current,reference,result_sha)
    assert result["bitwise_trajectories"]
    payload=torch.load(current/"baseline/sample.pt",weights_only=True)
    payload["latent"]=latent+1
    torch.save(payload,current/"baseline/sample.pt")
    manifest=json.loads((current/"baseline/complete.json").read_text())
    manifest["artifact_hashes"]["sample.pt"]=file_sha256(current/"baseline/sample.pt")
    write_json(current/"baseline/complete.json",manifest)
    with pytest.raises(RuntimeError,match="not bitwise"):
        verify_baseline_reference(current,reference,result_sha)


def test_full_unet_scope_enables_base_weights_and_keeps_other_models_frozen():
    unet=torch.nn.Linear(3,2)
    pipe=SimpleNamespace(unet=unet,vae=torch.nn.Linear(2,2).eval().requires_grad_(False),
                         text_encoder=torch.nn.Linear(2,2).eval().requires_grad_(False))
    reader=torch.nn.Linear(2,2).eval().requires_grad_(False)
    versions=frozen_versions(pipe,reader)
    assert len(trainable_unet_parameters(unet,"full_unet"))==2
    with torch.no_grad():
        unet.weight.add_(.01)
    frozen_audit(pipe,reader,versions,scope="full_unet")
    with pytest.raises(RuntimeError):
        trainable_unet_parameters(unet,"lora")
    unet.bias.requires_grad_(False)
    with pytest.raises(RuntimeError,match="every base weight"):
        trainable_unet_parameters(unet,"full_unet")
    unet.bias.requires_grad_(True)
    with torch.no_grad():
        pipe.vae.weight.add_(.01)
    with pytest.raises(RuntimeError,match="frozen model"):
        frozen_audit(pipe,reader,versions,scope="full_unet")


def test_periodic_full_checkpoint_replays_unsaved_updates_with_exact_rng_and_optimizer(tmp_path):
    torch.manual_seed(81)
    module=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
    optimizer=torch.optim.AdamW(module.parameters(),lr=.005)
    path=tmp_path/"checkpoint.pt"
    manifest={"trainable_scope":"full_unet","checkpoint_interval":4,"steps":12}
    losses=[]
    for step in range(1,11):
        optimizer.zero_grad(set_to_none=True)
        x,y=torch.randn(4,3),torch.randn(4,2)
        loss=(module(x)-y).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if checkpoint_due(step,12,4):
            save_training_checkpoint(path,trainable_module=module,optimizer=optimizer,epoch=0,
                episode_cursor=step,optimizer_step=step,manifest=manifest,trainer_state={"loss":loss.item()})
    expected=copy.deepcopy(module.state_dict())
    expected_optimizer=copy.deepcopy(optimizer.state_dict())
    # Simulate losing updates9/10 after the periodic step8 checkpoint.
    replacement=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
    second=torch.optim.AdamW(replacement.parameters(),lr=.1)
    restored=load_training_checkpoint(path,trainable_module=replacement,optimizer=second,expected_manifest=manifest)
    assert restored["optimizer_step"]==8
    for i in range(8,10):
        second.zero_grad(set_to_none=True)
        x,y=torch.randn(4,3),torch.randn(4,2)
        loss=(replacement(x)-y).square().mean()
        assert loss.item()==losses[i]
        loss.backward()
        second.step()
    for k,value in replacement.state_dict().items():
        torch.testing.assert_close(value,expected[k],rtol=0,atol=0)
    assert second.state_dict()["param_groups"]==expected_optimizer["param_groups"]
    for i,state in second.state_dict()["state"].items():
        for key,value in state.items():
            torch.testing.assert_close(value,expected_optimizer["state"][i][key],rtol=0,atol=0)


def test_checkpoint_schedule_preserves_first_final_and_graceful_stop():
    assert [s for s in range(1,13) if checkpoint_due(s,12,4)]==[1,4,8,12]
    assert checkpoint_due(7,12,4,stopping=True)
    assert all(checkpoint_due(s,12,1) for s in range(1,13))
    with pytest.raises(ValueError):
        checkpoint_due(1,12,0)

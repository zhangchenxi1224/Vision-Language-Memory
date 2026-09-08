import pytest
from scripts.inspire.run_oracle_to_unet_pipeline import trainer_command,validate_allocation,lease_remaining,snapshot_environment
from vision_memory.training.latent_teacher_bank import AuditError,write_json


def test_trainer_resume_uses_existing_latest_checkpoint_and_fixed_bank(tmp_path):
    config={"expected_commit":"a"*40,"trainer":{"dreamlite":"/models/dreamlite","reader_model":"/models/reader"}}
    command=trainer_command(config,tmp_path/"bank.json",tmp_path,12345)
    assert "--resume" not in command
    assert command[command.index("--deadline-unix")+1] == "12345"
    (tmp_path/"checkpoint-latest.pt").write_bytes(b"checkpoint marker only")
    assert "--resume" in trainer_command(config,tmp_path/"bank.json",tmp_path,12345)


def test_allocation_identity_requires_same_host_and_exact_gpu_uuids(monkeypatch):
    monkeypatch.setattr("scripts.inspire.run_oracle_to_unet_pipeline.socket.gethostname",lambda:"host-a")
    runtime={"expected_hostname":"host-a","gpu_indices":[0,1],"gpu_uuids":["GPU-A","GPU-B"]}
    assert validate_allocation(runtime,{0:"GPU-A",1:"GPU-B"}) == {"GPU-A","GPU-B"}
    with pytest.raises(AuditError,match="ownership"):
        validate_allocation(runtime,{0:"GPU-C",1:"GPU-B"})
    runtime["expected_hostname"]="host-b"
    with pytest.raises(AuditError,match="host"):
        validate_allocation(runtime,{0:"GPU-A",1:"GPU-B"})


def test_resource_extension_is_explicit_separate_lease_not_config_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr("scripts.inspire.run_oracle_to_unet_pipeline.time.time",lambda:1000)
    config={"resource_lease_path":str(tmp_path/"lease.json"),"runtime":{"expected_hostname":"host-a","gpu_uuids":["GPU-A","GPU-B"]}}
    lease={"expected_hostname":"host-a","gpu_uuids":["GPU-A","GPU-B"],"deadline_epoch":1100}
    write_json(tmp_path/"lease.json",lease)
    assert lease_remaining(config)[0] == 100
    lease["deadline_epoch"]=10000
    write_json(tmp_path/"lease.json",lease)
    assert lease_remaining(config)[0] == 9000
    lease["expected_hostname"]="wrong"
    write_json(tmp_path/"lease.json",lease)
    with pytest.raises(AuditError,match="lease"):
        lease_remaining(config)


def test_child_model_pins_come_from_sealed_bank_not_ambient_environment():
    bank={"models":{"dreamlite_mobile":{"passed":True,"manifest_sha256":"a"*64},
                    "qwen_reader":{"passed":True,"manifest_sha256":"b"*64}}}
    assert snapshot_environment(bank) == {"VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256":"a"*64,
                                         "VLM_READER_SNAPSHOT_MANIFEST_SHA256":"b"*64}
    bank["models"]["qwen_reader"]["passed"]=False
    with pytest.raises(AuditError,match="manifest SHA"):
        snapshot_environment(bank)

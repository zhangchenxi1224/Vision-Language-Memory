import hashlib
import json
from types import SimpleNamespace
import pytest
import torch
from vision_memory.repro.hf_snapshot import inspect_download,verify_download_seal
from vision_memory.dreamlite.native_base import NativeBaseEditSampler


def test_snapshot_seal_checks_hf_blob_hash_and_rejects_tampering(tmp_path):
    root=tmp_path/"model"
    root.mkdir()
    content=b'{"_class_name":"TestPipeline"}'
    (root/"model_index.json").write_bytes(content)
    metadata=root/".cache/huggingface/download/model_index.json.metadata"
    metadata.parent.mkdir(parents=True)
    etag=hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
    metadata.write_text("revision\n"+etag+"\n0\n")
    result=inspect_download(root,"revision")
    seal=tmp_path/"seal.json"
    seal.write_text(json.dumps(result))
    assert verify_download_seal(seal,root)==result
    (root/"model_index.json").write_bytes(content+b" ")
    with pytest.raises(ValueError,match="content hash"):
        verify_download_seal(seal,root)


def test_valid_hf_config_hash_does_not_make_a_missing_weight_snapshot_complete(tmp_path):
    payload=b'{"text_encoder":["transformers","Qwen3VLForConditionalGeneration"]}'
    (tmp_path/"model_index.json").write_bytes(payload)
    metadata=tmp_path/".cache/huggingface/download/model_index.json.metadata"
    metadata.parent.mkdir(parents=True)
    etag=hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()
    metadata.write_text("revision\n"+etag+"\n0\n")
    with pytest.raises(ValueError,match="Incomplete declared pipeline component"):
        inspect_download(tmp_path,"revision")


class FakeNativePipeline:
    def __init__(self):
        self.scheduler=SimpleNamespace(step=lambda *a,**k:(torch.ones(1,4,2,2),),sigmas=torch.tensor([1.,.5,0.]))
        self.prepare_latents=lambda: torch.zeros(1,4,2,2)
        self.prepare_image_latents=lambda: torch.zeros(1,4,2,2)
    def __call__(self,**kwargs):
        self.kwargs=kwargs
        self.initial=self.prepare_latents()
        self.prepare_image_latents()
        self.scheduler.step()
        output=self.scheduler.step()[0]
        return SimpleNamespace(images=output)


def test_native_base_keeps_official_pipeline_cfg_and_restores_hooks_on_failure():
    pipe=FakeNativePipeline()
    original=(pipe.prepare_latents,pipe.prepare_image_latents,pipe.scheduler.step)
    image=object()
    sampler=NativeBaseEditSampler(pipe,source_image=image,event_text="change the room",num_steps=2)
    noise=torch.randn(1,4,2,2)
    result=sampler(source_latents=torch.zeros_like(noise),noise_latents=noise,num_steps=2)
    assert pipe.kwargs["image"] is image
    assert pipe.kwargs["prompt"]=="change the room"
    assert pipe.kwargs["guidance_scale"]==7.5
    torch.testing.assert_close(result.trajectory[0],noise,rtol=0,atol=0)
    assert original==(pipe.prepare_latents,pipe.prepare_image_latents,pipe.scheduler.step)
    with pytest.raises(RuntimeError,match="source encoding"):
        sampler(source_latents=torch.ones_like(noise),noise_latents=noise,num_steps=2)
    assert original==(pipe.prepare_latents,pipe.prepare_image_latents,pipe.scheduler.step)

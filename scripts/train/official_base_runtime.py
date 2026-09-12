"""The pinned upstream DreamLitePipelineLoRA with verified Mobile-bank targets."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch


def audit_inference_only_runtime(runtime, frozen_versions):
    """An inference probe has no trainable adapter; audit that contract directly."""
    from scripts.train.train_latent_bank_unet import frozen_versions as versions
    pipe, reader = runtime["pipe"], runtime["reader"]
    for module in (pipe.unet, pipe.vae, pipe.text_encoder, reader):
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError("Inference-only parameter became trainable or acquired a gradient")
    if versions(pipe, reader) != frozen_versions:
        raise RuntimeError("Inference-only model parameters changed")


@torch.no_grad()
def load_base_runtime(args, bank, *, inference_only=False):
    from PIL import Image
    from peft import LoraConfig,get_peft_model
    from scripts.train import train_latent_bank_unet as training
    from scripts.train import r11_new_frozen_dreamlite_oracle as legacy
    from scripts.experiments.run_r11_open_answer_replay import snapshot_bindings
    from vision_memory.repro.hf_snapshot import verify_download_seal
    from vision_memory.repro import canonical_tensor_sha256
    from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler,freeze_module
    from vision_memory.dreamlite.conditioning import encode_image_edit_condition
    from vision_memory.dreamlite.differentiable_mobile import calculate_shift
    from vision_memory.dreamlite.native_base import NativeBaseEditSampler
    from vision_memory.dreamlite.source_images import load_sealed_source_image
    from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
    from vision_memory.training.latent_bank_unet import OFFICIAL_REFERENCE_COMMIT,file_sha256
    from vision_memory.reader.open_eos import assistant_termination_contract
    if not all((args.teacher_dreamlite,args.official_source,args.base_manifest)):
        raise ValueError("Base training requires teacher Mobile snapshot, pinned official source and sealed base snapshot")
    if args.flow_protocol!="official" or args.prompt_style!="official_raw":
        raise ValueError("Base arm implements the official raw-prompt FM training protocol")
    source_root=args.official_source.resolve()
    def verify_source():
        if subprocess.check_output(["git","rev-parse","HEAD"],cwd=source_root,text=True).strip()!=OFFICIAL_REFERENCE_COMMIT:
            raise ValueError("Wrong upstream DreamLite source")
        if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=source_root,text=True).strip():
            raise ValueError("Modified upstream DreamLite source")
    verify_source()
    base_seal=verify_download_seal(args.base_manifest,args.dreamlite)
    if base_seal["revision"]!="a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa":
        raise ValueError("Unexpected base revision")
    teacher_args=copy.copy(args)
    teacher_args.dreamlite=args.teacher_dreamlite
    teacher_args.reader=args.reader_model
    snapshots=snapshot_bindings(teacher_args)
    identities=lambda values:{(v["repo_id"],v["revision"],v["snapshot_payload_sha256"]) for v in values.values()}
    if identities(snapshots)!=identities(bank["snapshots"]):
        raise ValueError("Teacher/Reader bindings changed")
    # Reusing raw targets requires the exact same VAE, not an assumption that
    # two pipelines called DreamLite have the same latent coordinates.
    vae_hashes={}
    for component in ("config.json","diffusion_pytorch_model.safetensors"):
        base=args.dreamlite/"vae"/component
        mobile=args.teacher_dreamlite/"vae"/component
        if component.endswith("safetensors"):
            if file_sha256(base)!=file_sha256(mobile):
                raise ValueError("Base and teacher Mobile VAE weights differ")
            vae_hashes[component]=file_sha256(base)
        else:
            left,right=(json.loads(p.read_text()) for p in (base,mobile))
            normalize=lambda config:{k:v for k,v in config.items() if not k.startswith("_")}
            if normalize(left)!=normalize(right):
                raise ValueError("Base and teacher VAE architectures/latent conventions differ")
    sys.path.insert(0,str(source_root))
    from dreamlite import DreamLitePipelineLoRA
    vd,rd=torch.device(args.dreamlite_device),torch.device(args.reader_device)
    colocated=bool(getattr(args,"colocate_models",False))
    if vd.type!="cuda" or rd.type!="cuda" or (vd==rd and not (inference_only or colocated)):
        raise ValueError("Base Writer and Reader require separate CUDA devices")
    pipe=DreamLitePipelineLoRA.from_pretrained(args.dreamlite,local_files_only=True,torch_dtype=torch.float32).to(vd)
    for module in (pipe.unet,pipe.vae,pipe.text_encoder): freeze_module(module)
    processor,reader=legacy._load_reader(teacher_args,rd,torch.bfloat16)
    torch.manual_seed(args.seed)
    scope=getattr(args,"trainable_scope","lora")
    if scope=="lora":
        pipe.unet=get_peft_model(pipe.unet,LoraConfig(r=args.lora_rank,lora_alpha=args.lora_rank,lora_dropout=0.,
                                                   target_modules=["to_q","to_k","to_v","to_out.0"]))
    elif scope=="full_unet":
        pipe.unet.requires_grad_(True)
    else:
        raise ValueError("Unknown U-Net trainable scope")
    pipe.unet.eval()
    if inference_only:
        # B=0 leaves the pretrained model unchanged; no optimizer is permitted
        # in read-only probes using the otherwise idle Reader GPU.
        freeze_module(pipe.unet)
    predictor=DifferentiableDreamLiteMobileSampler.from_pipeline(pipe,checkpoint_unet=False)
    contexts={}
    image=Image.new("RGB",(1024,1024),(128,128,128))
    image_tensor=pipe.image_processor.preprocess(image)
    gray_source=pipe.prepare_image_latents(image_tensor,dtype=torch.float32,device=vd)
    # Training uses the same source image encoding as official inference.
    old_gray=legacy.encode_model_latent(pipe.vae,legacy.blank_source_rgb(device=vd,dtype=torch.float32))
    sigmas=np.linspace(1.,1./28,28)
    config=pipe.scheduler.config
    mu=calculate_shift(gray_source.shape[-2]*gray_source.shape[-1]//4,config.get("base_image_seq_len",256),
        config.get("max_image_seq_len",4096),config.get("base_shift",.5),config.get("max_shift",1.16))
    pipe.scheduler.set_timesteps(sigmas=sigmas,device=vd,mu=mu)
    effective=tuple(float(x) for x in pipe.scheduler.sigmas[:28].cpu())
    source_bindings={}
    source_image_files={}
    gray_image=image
    blank_pixels=decode_model_latents_unit_interval(pipe.vae,gray_source,clamp=True).cpu()
    for group in bank["groups"]:
        bank_source=training.resolve_payload(group,"source_latent",args.bank_manifest).to(vd)
        if group.get("source_kind")=="blank_gray_1024":
            image,source=gray_image,gray_source
            if not torch.equal(bank_source,old_gray): raise ValueError("Bank source cannot be replayed with base VAE")
            source_reason="official source PIL gray128 preprocessing instead of historical exact float gray0.5"
        elif group.get("source_kind")=="sealed_rgb_1024":
            image,image_path=load_sealed_source_image(group,args.bank_manifest)
            source_image_files[image_path]=group["source_image_file_sha256"]
            source=pipe.prepare_image_latents(pipe.image_processor.preprocess(image),dtype=torch.float32,device=vd)
            if not torch.equal(bank_source,source):
                raise ValueError("Sealed source latent differs from official RGB image encoding")
            source_reason="exact sealed RGB PNG encoded by the official Base VAE for training and native inference"
        else:
            raise ValueError("Unsupported source kind; source provenance must be explicit")
        if source.shape!=gray_source.shape:
            raise ValueError("Source resolution differs from the fixed official schedule")
        condition=encode_image_edit_condition(pipe,image,group["event_text"],device=vd,dtype=torch.float32)
        donor=group.get("donor_control",bank.get("donor_control",{}))
        if not donor or donor.get("answer","").casefold()==group["answer"].casefold():
            raise ValueError("Different-answer donor required")
        if "image_path" in donor:
            donor_pixels=training.resolve_payload(donor,"image",args.bank_manifest)
        else:
            donor_pixels=decode_model_latents_unit_interval(pipe.vae,
                training.resolve_payload(donor,"latent",args.bank_manifest).to(vd),clamp=True).cpu()
        if donor_pixels.ndim==3: donor_pixels=donor_pixels.unsqueeze(0)
        qid=group["question_id"]
        contexts[qid]={"source":source,"condition":condition,"effective_sigmas":effective,"num_inference_steps":28,
            "inference_sampler":NativeBaseEditSampler(pipe,source_image=image,event_text=group["event_text"]),
            "blank":blank_pixels,
            "donor":donor_pixels,"donor_answer":donor["answer"]}
        source_bindings[qid]={"bank_source_sha256":canonical_tensor_sha256(bank_source.cpu()),
            "official_source_sha256":canonical_tensor_sha256(source.cpu()),
            "rms_difference":float((source-bank_source).double().square().mean().sqrt()),
            "reason":source_reason}
        if group["source_kind"]=="sealed_rgb_1024":
            source_bindings[qid]["source_image_file_sha256"]=group["source_image_file_sha256"]
    def verify_extra():
        verify_source()
        if verify_download_seal(args.base_manifest,args.dreamlite)!=base_seal:
            raise ValueError("Base snapshot changed")
        for path,sha in source_image_files.items():
            if file_sha256(path)!=sha:
                raise ValueError("Sealed source image changed during execution")
    return dict(pipe=pipe,reader=reader,processor=processor,sampler=predictor,contexts=contexts,
        vae_device=vd,reader_device=rd,snapshots=snapshots,termination=assistant_termination_contract(reader,processor),
        protocol_binding={"student":"official DreamLitePipelineLoRA base","base_snapshot":base_seal,
            "official_source_commit":OFFICIAL_REFERENCE_COMMIT,"vae_weights_sha256":vae_hashes,"source_bindings":source_bindings,
            "train_prompt":"raw event, upstream LoRA example","inference_prompt":"native upstream diptych+CFG",
            "inference_guidance_scale":7.5,"inference_image_guidance_scale":1.,"inference_steps":28},
        verify_additional_bindings=verify_extra)

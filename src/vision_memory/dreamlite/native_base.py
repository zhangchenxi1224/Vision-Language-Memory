"""Evaluation adapter that delegates all denoising/CFG to official base code."""
from __future__ import annotations
import torch
from .differentiable_mobile import DreamLiteSamplerOutput


class NativeBaseEditSampler:
    def __init__(self, pipeline, *, source_image, event_text, num_steps=28, guidance_scale=7.5,
                 inference_condition="native"):
        self.pipeline = pipeline
        self.source_image = source_image
        self.event_text = event_text
        self.num_steps = num_steps
        if not 1.0 <= guidance_scale <= 100.0:
            raise ValueError("Invalid native base guidance scale")
        self.guidance_scale = float(guidance_scale)
        if inference_condition not in ("native", "training_raw"):
            raise ValueError("Unknown Base inference condition")
        if inference_condition == "training_raw" and self.guidance_scale != 1.:
            raise ValueError("Raw training-condition control requires guidance_scale=1")
        self.inference_condition = inference_condition

    @torch.no_grad()
    def __call__(self, *, source_latents, noise_latents, num_steps, return_trajectory=True, **_):
        if num_steps != self.num_steps:
            raise ValueError("Base inference step count changed")
        pipe=self.pipeline
        original_noise=pipe.prepare_latents
        original_source=pipe.prepare_image_latents
        original_step=pipe.scheduler.step
        original_encode=pipe.encode_prompt if self.inference_condition == "training_raw" else None
        raw_calls=[]
        if original_encode is not None:
            from .conditioning import encode_image_edit_condition
            # Re-encode the actual current RGB source/event for every write.
            # This reproduces the upstream LoRA training encoder batch, without
            # consulting a development bank or retaining an answer/state ledger.
            condition=encode_image_edit_condition(pipe,self.source_image,self.event_text,
                device=noise_latents.device,dtype=noise_latents.dtype,prompt_style="official_raw")
            if (condition.prompt_embeds.ndim!=3 or condition.prompt_embeds.shape[0]!=1
                    or condition.attention_mask.shape!=condition.prompt_embeds.shape[:2]):
                raise ValueError("Unexpected raw-event encoder batch")
            def raw_encode(*args, **kwargs):
                if args or kwargs.get("mode")!="edit" or len(kwargs.get("prompts",[]))!=3:
                    raise RuntimeError("Unexpected native conditioning call")
                raw_calls.append(True)
                return condition.prompt_embeds.repeat(3,1,1),condition.attention_mask.repeat(3,1)
        path=[noise_latents.detach().clone()]
        seen=[False,False]

        def fixed_noise(*args, **kwargs):
            seen[0]=True
            return noise_latents.clone()

        def checked_source(*args, **kwargs):
            value=original_source(*args, **kwargs)
            if not torch.equal(value,source_latents):
                raise RuntimeError("Official base source encoding differs between training and inference")
            seen[1]=True
            return value

        def capture_step(*args, **kwargs):
            value=original_step(*args, **kwargs)
            path.append(value[0].detach().clone())
            return value

        pipe.prepare_latents=fixed_noise
        pipe.prepare_image_latents=checked_source
        pipe.scheduler.step=capture_step
        if original_encode is not None:
            pipe.encode_prompt=raw_encode
        try:
            output=pipe(prompt=self.event_text,image=self.source_image,num_inference_steps=num_steps,
                        guidance_scale=self.guidance_scale,image_guidance_scale=1.0,output_type="latent").images
        finally:
            pipe.prepare_latents=original_noise
            pipe.prepare_image_latents=original_source
            pipe.scheduler.step=original_step
            if original_encode is not None:
                pipe.encode_prompt=original_encode
        if original_encode is not None and len(raw_calls)!=1:
            raise RuntimeError("Native sampler did not consume the raw condition exactly once")
        if not all(seen) or len(path)!=num_steps+1 or not torch.equal(output,path[-1]):
            raise RuntimeError("Failed to capture the native base generation path")
        sigmas=tuple(float(x) for x in pipe.scheduler.sigmas[:num_steps].cpu())
        if abs(sigmas[0]-1.)>2e-6 or abs(float(pipe.scheduler.sigmas[-1]))>2e-6:
            raise RuntimeError("Native base did not integrate the complete noise-to-target schedule")
        return DreamLiteSamplerOutput(output,tuple(path) if return_trajectory else None,sigmas)

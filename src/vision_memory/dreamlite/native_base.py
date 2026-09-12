"""Evaluation adapter that delegates all denoising/CFG to official base code."""
from __future__ import annotations
import torch
from .differentiable_mobile import DreamLiteSamplerOutput


class NativeBaseEditSampler:
    def __init__(self, pipeline, *, source_image, event_text, num_steps=28, guidance_scale=7.5):
        self.pipeline = pipeline
        self.source_image = source_image
        self.event_text = event_text
        self.num_steps = num_steps
        if not 1.0 <= guidance_scale <= 100.0:
            raise ValueError("Invalid native base guidance scale")
        self.guidance_scale = float(guidance_scale)

    @torch.no_grad()
    def __call__(self, *, source_latents, noise_latents, num_steps, return_trajectory=True, **_):
        if num_steps != self.num_steps:
            raise ValueError("Base inference step count changed")
        pipe=self.pipeline
        original_noise=pipe.prepare_latents
        original_source=pipe.prepare_image_latents
        original_step=pipe.scheduler.step
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
        try:
            output=pipe(prompt=self.event_text,image=self.source_image,num_inference_steps=num_steps,
                        guidance_scale=self.guidance_scale,image_guidance_scale=1.0,output_type="latent").images
        finally:
            pipe.prepare_latents=original_noise
            pipe.prepare_image_latents=original_source
            pipe.scheduler.step=original_step
        if not all(seen) or len(path)!=num_steps+1 or not torch.equal(output,path[-1]):
            raise RuntimeError("Failed to capture the native base generation path")
        sigmas=tuple(float(x) for x in pipe.scheduler.sigmas[:num_steps].cpu())
        if abs(sigmas[0]-1.)>2e-6 or abs(float(pipe.scheduler.sigmas[-1]))>2e-6:
            raise RuntimeError("Native base did not integrate the complete noise-to-target schedule")
        return DreamLiteSamplerOutput(output,tuple(path) if return_trajectory else None,sigmas)

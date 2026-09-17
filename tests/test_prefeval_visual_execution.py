import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from scripts.experiments.build_prefeval_rgb_teachers import validate_vae,check_completed,schedule
from scripts.eval.prefeval_rgb_baseline import write_from_png,pixels_sha

class VisualExecutionTests(unittest.TestCase):
    def test_wrong_vae_rejected(self):
        config=SimpleNamespace(latent_channels=4,scaling_factor=1.,shift_factor=0.)
        tiny=type('AutoencoderTiny',(),{} )();tiny.dtype=torch.float32;tiny.config=config
        validate_vae(tiny)
        with self.assertRaises(ValueError):validate_vae(SimpleNamespace(dtype=torch.float32,config=config))
        tiny.config.scaling_factor=.18215
        with self.assertRaises(ValueError):validate_vae(tiny)

    def test_completion_cannot_hide_mismatched_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'result.json').write_text(json.dumps(dict(binding={'old':1})))
            with self.assertRaises(ValueError):check_completed(out,{'new':2})

    def test_registered_sentinel_has_all_initializer_dependencies(self):
        p=Path(__file__).resolve().parents[1]/'reports/prefeval-rgb-20260917/registered/manifest.json'
        m=json.loads(p.read_text(encoding='utf-8'));lanes=schedule(m['targets'],m['sentinel_targets'],4)
        self.assertEqual(set(sum(lanes,[])),set(m['sentinel_targets']))
        for lane in lanes:
            seen=set()
            for sid in lane:
                pred=m['targets'][sid]['predecessor']
                self.assertTrue(pred is None or pred in seen);seen.add(sid)

    def test_png_only_boundary_and_repeated_state_still_writes(self):
        calls=[]
        class Memory:
            def __init__(self,pipe,*,image,guidance_scale,inference_condition):self.image=image
            def write(self,event,*,seed):
                calls.append((pixels_sha(self.image),event,seed))
                out=self.image.copy();out.putpixel((0,0),(seed,2,3));z=torch.zeros(1,4,128,128)
                return SimpleNamespace(image=out,source_latent=z,noise=z,trajectory=[z]*28)
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.eval.prefeval_rgb_baseline.OfficialRGBMemory',Memory):
            root=Path(tmp);Image.new('RGB',(1024,1024),(128,128,128)).save(root/'a.png')
            a=write_from_png(None,root/'a.png','retain',1,root/'b.png')
            b=write_from_png(None,root/'b.png','retain',2,root/'c.png')
            self.assertEqual(a['output_pixels_sha'],b['source_pixels_sha'])
            self.assertEqual(len(calls),2)
            self.assertEqual(a['model_inputs'],['previous_png','event','external_noise'])

if __name__=='__main__':unittest.main()

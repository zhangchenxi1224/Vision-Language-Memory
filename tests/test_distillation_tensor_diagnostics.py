import importlib.util
import io
from pathlib import Path
import unittest
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/'analysis'/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

export=module('export_distillation_tensor_bundle')
analyze=module('analyze_distillation_tensor_bundle')


class TensorDiagnosticsTest(unittest.TestCase):
    def test_export_preserves_view_offset_and_stride(self):
        original=torch.arange(60,dtype=torch.float32).reshape(5,12)[1:4,2:10:2].T
        payload=io.BytesIO();torch.save({'latent':original,'image':torch.ones(1,3,8,8)},payload)
        with zipfile.ZipFile(payload) as archive:
            name=next(n for n in archive.namelist() if n.endswith('/data.pkl'))
            spec=export.MetadataReader(io.BytesIO(archive.read(name))).load()['latent']
            raw=archive.read(name[:-len('data.pkl')]+'data/'+spec['storage']['key'])
            storage=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy())
            restored=torch.as_strided(storage,spec['shape'],spec['stride'],storage_offset=spec['offset'])
            self.assertTrue(torch.equal(restored,original))

    def test_spectral_control_separates_missing_high_frequency(self):
        x=torch.arange(32,dtype=torch.float64)
        low=torch.cos(2*torch.pi*x/32)[None,:].repeat(32,1)
        high=(-1.)**(x[:,None]+x[None,:])
        result=analyze.spectrum_fit(low,low+high)
        self.assertAlmostEqual(result['high']['error_energy_fraction'],1.,places=10)
        self.assertAlmostEqual(result['high']['error_to_target_energy'],1.,places=10)
        self.assertAlmostEqual(result['low']['error_to_target_energy'],0.,places=10)


if __name__=='__main__': unittest.main()

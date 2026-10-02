import ast
import inspect
import copy
import pytest
import torch
from scripts.probes import prefeval_rgb_endpoint_diagnostic as d
from scripts.reporting.verify_prefeval_rgb_endpoint_diagnostic import exact_index

def test_representation_and_device_values_are_isolated():
    decoded=torch.tensor([.121,.451,.871],dtype=torch.float32)[:,None,None].expand(3,2,2).clone()
    png=decoded.mul(255).round()/255
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    images=d.matched_images(decoded.to(device),png,device)
    assert torch.equal(images['float_cpu'],decoded)
    assert torch.equal(images['png_cpu'],png)
    assert torch.equal(images['float_cuda'].cpu(),decoded)
    assert torch.equal(images['png_cuda'].cpu(),png)
    assert len({x.stride() for x in images.values()})==1

def test_reconstruction_drift_does_not_modify_archived_pixels():
    png=torch.full((3,2,2),128/255);saved=png.clone()
    with pytest.raises(ValueError,match='reconstruction'):
        d.matched_images(torch.zeros_like(png),png,'cpu')
    assert torch.equal(png,saved)

def test_query_partitions_and_all_registered_coverage():
    m=d.load_json(d.REPORT/'registered/manifest.json');o=d.load_json(d.REPORT/'reader-format-v2.json')
    total=0
    for sid in m['sentinel_targets']:
        qs=d.recovery_queries(o,sid);assert len(qs)==len({q['query_id'] for q in qs})
        assert len(qs)==5*len(m['targets'][sid]['state']);total+=len(qs)
        for scope in m['targets'][sid]['state']:
            assert {q['partition'] for q in qs if q['scope']==scope}=={'training','heldout'}
    assert total*4==1760

def test_missing_and_duplicate_cells_fail():
    key=lambda r:r['id'];rows=[{'id':'a'},{'id':'b'}]
    assert len(exact_index(rows,key,{'a','b'}))==2
    for bad in (rows[:1],rows+rows[:1],rows+[{'id':'c'}]):
        with pytest.raises(ValueError):exact_index(bad,key,{'a','b'})

def test_current_value_mcq_lineage_retains_all_occurrences():
    m=d.load_json(d.REPORT/'registered/manifest.json');o=d.load_json(d.REPORT/'reader-format-v2.json')
    count=0;overwritten=0;cleared=0
    for sid in m['sentinel_targets']:
        t=m['targets'][sid];qs=d.application_queries(m,o,sid);count+=len(qs)
        assert {q['scope'] for q in qs}=={s for s,v in t['state'].items() if v is not None}
        for q in qs:
            r=m['records'][q['record']]
            assert r['preference']==t['state'][q['scope']]==q['value']
            assert q['target_index']==r['target_index'] and q['options']==r['options']
            if t['predecessor']:
                old=m['targets'][t['predecessor']]['state'].get(q['scope'])
                overwritten+=old is not None and old!=q['value']
        cleared+=sum(v is None for v in t['state'].values())
    assert count==84 and overwritten>0 and cleared==4

def test_missing_current_mcq_cannot_fall_back_to_stale_label():
    m=d.load_json(d.REPORT/'registered/manifest.json');o=d.load_json(d.REPORT/'reader-format-v2.json')
    sid=m['sentinel_targets'][0];scope=next(iter(m['targets'][sid]['state']))
    bad=copy.deepcopy(m);bad['targets'][sid]['state'][scope]='unregistered replacement'
    with pytest.raises(ValueError,match='current-value'):d.application_queries(bad,o,sid)

def test_generation_boundary_does_not_accept_target_or_state():
    assert set(inspect.signature(d.generate).parameters)=={'reader','processor','image','query','device'}
    tree=ast.parse(inspect.getsource(d))
    forbidden=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr in ('backward','step','Adam','Parameter')]
    assert not forbidden

def test_prefix_comparison_detects_visual_and_token_drift():
    gen={'input_ids':torch.tensor([[1,2]]),'pixel_values':torch.ones(2,3),'mm_token_type_ids':torch.tensor([[1,0]])}
    ce={'input_ids':torch.tensor([[1,2,3]]),'pixel_values':torch.ones(2,3),'mm_token_type_ids':torch.tensor([[1,0,0]])}
    d.compare_prefix(gen,ce,1)
    ce['pixel_values'][0,0]=2
    with pytest.raises(ValueError,match='pixel_values'):d.compare_prefix(gen,ce,1)

def test_capture_processor_preserves_inputs_outputs_and_grad_flags():
    class Fake:
        def __call__(self,**kw):return {'pixel_values':kw['images'][0]}
    p=d.CaptureProcessor(Fake());x=torch.ones(3,2,2)
    output=p(images=[x]);assert output['pixel_values'] is x and not x.requires_grad
    assert torch.equal(p.last['pixel_values'],x) and p.last['pixel_values'] is not x

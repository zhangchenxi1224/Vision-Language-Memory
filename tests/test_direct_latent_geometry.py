import numpy as np
import pytest
from vision_memory.training.direct_latent_geometry import draw, initial_array, panel, study_members, spectrum


def test_fixed_panel_reuse_does_not_duplicate_starts():
    rows = panel()
    assert len(rows) == len({r['run_id'] for r in rows}) == 96
    for distribution in ['gaussian','uniform','sphere','rademacher','heavy_tail']:
        assert len(study_members(rows, 'distribution', distribution)) == 8
    for scale in [.25,.5,1.,2.,4.]:
        assert len(study_members(rows, 'scale', scale)) == 8
    assert len(study_members(rows, 'gaussian_density', None)) == 32
    assert not set(r['run_id'] for r in rows[::2]) & set(r['run_id'] for r in rows[1::2])


def test_shapes_and_moments_remain_distinguishable():
    shape = (1,4,128,128)
    g = draw(shape,'gaussian',0)
    s = draw(shape,'sphere',0)
    np.testing.assert_array_equal(g,draw(shape,'gaussian',0))
    assert not np.array_equal(g,s)
    assert np.sqrt(np.mean(s*s)) == pytest.approx(1.,abs=1e-12)
    assert set(np.unique(draw(shape,'rademacher',0))) == {-1.,1.}
    assert abs(draw(shape,'uniform',0)).max() <= np.sqrt(3)
    t=draw(shape,'heavy_tail',0)
    assert np.mean(t**4)/np.mean(t*t)**2 > 4


def test_scale_reuses_direction_and_preserves_shared_center():
    ref=np.linspace(-1,1,64).reshape(1,4,4,4).astype('float32')
    spec={'distribution':'gaussian','seed':2,'scale':1}
    a,_=initial_array(ref,spec)
    b,_=initial_array(ref,{**spec,'scale':2})
    np.testing.assert_allclose(b-ref,2*(a-ref),atol=1e-7)
    assert a.dtype==np.float32
    with pytest.raises(ValueError):initial_array(ref,{**spec,'scale':float('nan')})


def test_rank_report_does_not_claim_dimension_from_sample_count():
    out=spectrum(np.random.default_rng(0).standard_normal((8,500)))
    assert out['rank_cap']==7 and 1<=out['r95']<=7
    assert spectrum(np.ones((1,500)))['r95'] is None


def test_question_only_template_is_exact_and_answer_free():
    from vision_memory.training.direct_latent_geometry import question_prompts, INSTRUCTIONS
    prompts=question_prompts()
    assert len(prompts)==5 and len(set(prompts.values()))==5
    for text in prompts.values():
        assert text.split('\n')[1:]==INSTRUCTIONS.split('\n')
        assert len(text.split('\n'))==3
        assert 'ambient' not in text and 'Choose exactly one option' not in text
        assert 'indigo desk train 001123' in text and 'later check' in text and 'music' in text
    assert all(r['arm']=='open_gold_eos' for r in panel())

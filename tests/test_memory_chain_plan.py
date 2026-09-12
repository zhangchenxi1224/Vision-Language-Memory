from scripts.probes.official_memory_chains import chain_plan


def test_chains_cover_every_directed_overwrite_and_keep_noops_source_dependent():
    plan = chain_plan()
    assert plan == chain_plan() and len(plan) == 16
    edges = set()
    for sequence in plan:
        steps = sequence["steps"]
        assert len(steps) == 6
        for i in (1, 3, 5):
            assert steps[i]["operation"] == "noop"
            assert steps[i]["gold"] == steps[i - 1]["gold"]
            assert steps[i]["event_text"] == steps[1]["event_text"]
        states = [steps[i]["expected_state"] for i in (0, 2, 4)]
        edges.update(zip(states, states[1:]))
    assert edges == {(a, b) for a in ("ambient", "jazz", "clear") for b in ("ambient", "jazz", "clear") if a != b}
    assert len({step["noise_seed"] for seq in plan for step in seq["steps"]}) == 24
    for repetition in range(4):
        seeds = [tuple(step["noise_seed"] for step in seq["steps"]) for seq in plan if seq["repetition"] == repetition]
        assert len(set(seeds)) == 1


def test_cfg1_chains_keep_the_task_and_use_a_new_noise_namespace():
    old=chain_plan()
    new=chain_plan(guidance_scale=1.0)
    old_seeds={s['noise_seed'] for q in old for s in q['steps']}
    new_seeds={s['noise_seed'] for q in new for s in q['steps']}
    assert len(new_seeds)==24 and old_seeds.isdisjoint(new_seeds)
    for a,b in zip(old,new):
        assert (a['sequence'],a['repetition'])==(b['sequence'],b['repetition'])
        assert [{k:v for k,v in s.items() if k!='noise_seed'} for s in a['steps']]==[{k:v for k,v in s.items() if k!='noise_seed'} for s in b['steps']]

from vision_memory.training.learnability import evaluation_seed, gate, teacher_order, training_pair


def test_stage_pairs_isolate_noise_then_target_and_cover_bank():
    ids=["anchor", "b", "c"]
    a=[training_pair("single",i,ids,7) for i in range(9)]
    b=[training_pair("noise",i,ids,7) for i in range(9)]
    c=[training_pair("set",i,ids,7) for i in range(9)]
    assert len(set(a))==1 and len(set(b))==3 and len(set(c))==3
    assert all(t=="anchor" for _,t in b)
    assert [n for n,_ in b]==[n for n,_ in c]
    assert {t for _,t in c}==set(ids)
    assert {n for n,_ in c}.isdisjoint({evaluation_seed(7,s,i) for s in ("validation","test") for i in range(8)})


def row(split="train",correct=True,ratio=.01,prompt="original_open",eos=True):
    return {"split":split,"prompt_id":prompt,"relative_rms":ratio,
            "scorer":{"strict_correct":correct,"answer_followed_immediately_by_eos":eos}}


def test_gate_needs_raw_answer_stop_and_coordinate_fit():
    assert gate([row()],"single")["passed"]
    for r in [row(correct=False),row(eos=False),row(ratio=.3),row(ratio=float('nan'))]:
        assert not gate([r],"single")["passed"]
    assert gate([row(),row("test",False),row(prompt="paraphrase_4",correct=False)],"single")["passed"]


def test_validation_only_enters_later_gate_and_set_has_no_fake_label():
    rows=[row(),row("validation",False)]
    assert gate(rows,"single")["passed"]
    assert not gate(rows,"noise")["passed"]
    assert gate([row(),row("validation",ratio=999)],"set")["passed"]


def test_anchor_choice_ignores_score_and_input_order():
    b={"teachers":[{"teacher_id":"x","source_runs":[{"run_id":"z"}]},
                   {"teacher_id":"y","source_runs":[{"run_id":"direct-gaussian-s00-a1"}]}]}
    assert teacher_order(b)==["y","x"]
    b["teachers"].reverse()
    assert teacher_order(b)==["y","x"]

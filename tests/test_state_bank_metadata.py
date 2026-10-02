import copy
import pytest
from scripts.reporting.complete_state_bank_metadata import complete_bindings


def test_state_bank_metadata_completion_preserves_targets_and_binds_runtime_snapshots():
    parent={'models':{'vae':{'manifest_sha256':'bound'}},'snapshots':{'vae':{
        'repo_id':'model','revision':'fixed','snapshot_payload_sha256':'payload'}}}
    bank={'models':parent['models'],'provenance':{'parent_bank_sha256':'parent'},
          'teachers':[{'latent_path':'original.pt','latent_sha256':'same'}],'groups':[{'event_text':'jazz'}]}
    original=copy.deepcopy(bank)
    new=complete_bindings(bank,parent,'source','parent')
    assert bank==original
    assert new['teachers']==original['teachers'] and new['groups']==original['groups']
    runtime_identity=lambda v:{(x['repo_id'],x['revision'],x['snapshot_payload_sha256']) for x in v.values()}
    assert runtime_identity(new['snapshots'])==runtime_identity(parent['snapshots'])
    with pytest.raises(ValueError,match='only'):
        complete_bindings(new,parent,'source','parent')
    with pytest.raises(ValueError,match='do not match'):
        complete_bindings(bank,parent,'source','different-parent')

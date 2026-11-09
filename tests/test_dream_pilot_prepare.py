"""Pilot admission verifies whole objects and the exact selected line ranges."""
import copy
import hashlib

import pytest

from scripts.dream_pilot_prepare import verify
from sucoder import dream


def corpus(tmp_path):
    raw = b"outside admission\nselected passage\n"
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    (tmp_path / blob).write_bytes(raw)
    unit = dict(git_blob=blob, byte_start=18, byte_end=len(raw), bytes=17, lines=[2, 2], sha256=dream.digest(raw[18:]))
    sources = [dict(unit, unit_id="unit-" + str(index)) for index in range(6)]
    return dict(corpora=[dict(repository="sample", commit="a" * 40, selected_bytes=102, sources=sources)]), blob


def test_pilot_admission_reproduces_declared_passages(tmp_path):
    manifest, blob = corpus(tmp_path)
    result = verify(manifest, tmp_path)
    assert len(result[0]["units"]) == 6
    assert result[0]["units"][0]["original_commit"] == "a" * 40


def test_pilot_admission_rejects_changed_unselected_object_bytes(tmp_path):
    manifest, blob = corpus(tmp_path)
    path = tmp_path / blob
    path.write_bytes(path.read_bytes().replace(b"outside", b"altered"))
    with pytest.raises(dream.DreamError, match="Frozen pilot source differs"):
        verify(manifest, tmp_path)


@pytest.mark.parametrize("defect", ["range", "hash", "count"])
def test_pilot_admission_rejects_inconsistent_manifest(tmp_path, defect):
    original, blob = corpus(tmp_path)
    manifest = copy.deepcopy(original)
    if defect == "range":
        manifest["corpora"][0]["sources"][0]["lines"] = [1, 2]
    elif defect == "hash":
        manifest["corpora"][0]["sources"][0]["sha256"] = "b" * 64
    else:
        manifest["corpora"][0]["sources"].pop()
    with pytest.raises(dream.DreamError):
        verify(manifest, tmp_path)

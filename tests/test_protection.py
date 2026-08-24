"""Cancellable templates: recognition must survive the transform, breaches must not."""

import numpy as np
import pytest

from faceguard import ProtectedMatcher, TemplateProtector
from faceguard.recognition import FaceEmbedder
from faceguard.utils import synth_face

KEY = b"a-32-byte-deployment-secret-key!"


@pytest.fixture(scope="module")
def embeddings():
    embedder = FaceEmbedder()
    return [embedder.embed(synth_face(seed=s)) for s in range(4)]


def test_key_material_is_validated():
    with pytest.raises(ValueError):
        TemplateProtector(key=b"short")
    with pytest.raises(ValueError):
        TemplateProtector(key=KEY, bits=32)


def test_the_same_face_and_salt_always_yield_the_same_template(embeddings):
    p = TemplateProtector(KEY)
    assert np.array_equal(p.protect(embeddings[0], b"s"), p.protect(embeddings[0], b"s"))
    assert p.distance(p.protect(embeddings[0], b"s"), p.protect(embeddings[0], b"s")) == 0.0


def test_templates_preserve_the_distance_ordering_that_matching_depends_on(embeddings):
    p = TemplateProtector(KEY, bits=512)
    reference = embeddings[0]
    near = reference + 0.02 * np.random.default_rng(0).standard_normal(len(reference))
    far = -reference

    t_ref = p.protect(reference)
    assert p.distance(t_ref, p.protect(near)) < p.distance(t_ref, p.protect(far))


def test_hamming_recovers_the_underlying_cosine_distance(embeddings):
    p = TemplateProtector(KEY, bits=1024)
    a, b = embeddings[0], -embeddings[0]  # exactly opposed: cosine distance 2
    recovered = p.cosine_from_hamming(p.distance(p.protect(a), p.protect(b)))
    assert recovered == pytest.approx(2.0, abs=0.15)


def test_a_different_salt_makes_the_same_face_unlinkable(embeddings):
    """Two deployments enrolling the same person must not be joinable."""
    p = TemplateProtector(KEY)
    t1 = p.protect(embeddings[0], b"deployment-a")
    t2 = p.protect(embeddings[0], b"deployment-b")
    # ~half the bits differ: the templates look like unrelated people.
    assert 0.35 < p.distance(t1, t2) < 0.65


def test_a_different_key_invalidates_every_stored_template(embeddings):
    """Key rotation is what makes a breach recoverable."""
    old = TemplateProtector(KEY).protect(embeddings[0], b"s")
    new = TemplateProtector(b"a-completely-different-secret!!!!").protect(embeddings[0], b"s")
    assert TemplateProtector.distance(old, new) > 0.3


def test_reissue_replaces_a_compromised_template_without_re_enrolment(embeddings):
    p = TemplateProtector(KEY)
    compromised = p.protect(embeddings[0], b"old-salt")
    fresh = p.reissue(embeddings[0], b"new-salt")
    assert p.distance(compromised, fresh) > 0.3


def test_mismatched_template_lengths_are_an_error(embeddings):
    a = TemplateProtector(KEY, bits=128).protect(embeddings[0])
    b = TemplateProtector(KEY, bits=256).protect(embeddings[0])
    with pytest.raises(ValueError):
        TemplateProtector.distance(a, b)


def test_protected_gallery_identifies_its_enrolled_subjects(embeddings):
    matcher = ProtectedMatcher(TemplateProtector(KEY, bits=512))
    for i, e in enumerate(embeddings):
        matcher.enroll(f"subject-{i}", e)
    assert len(matcher) == len(embeddings)
    for i, e in enumerate(embeddings):
        name, distance = matcher.identify(e)
        assert name == f"subject-{i}" and distance == pytest.approx(0.0, abs=1e-9)


def test_protected_gallery_rejects_a_stranger(embeddings):
    matcher = ProtectedMatcher(TemplateProtector(KEY, bits=512), match_threshold=0.2)
    matcher.enroll("enrolled", embeddings[0])
    name, _ = matcher.identify(-embeddings[0])
    assert name is None


def test_revoke_actually_forgets_the_subject(embeddings):
    matcher = ProtectedMatcher(TemplateProtector(KEY))
    matcher.enroll("alice", embeddings[0])
    matcher.revoke("alice")
    assert len(matcher) == 0 and matcher.identify(embeddings[0])[0] is None


def test_rotate_keeps_the_subject_matchable_under_a_fresh_salt(embeddings):
    matcher = ProtectedMatcher(TemplateProtector(KEY, bits=512))
    matcher.enroll("alice", embeddings[0])
    matcher.rotate("alice", embeddings[0])
    assert matcher.identify(embeddings[0])[0] == "alice"

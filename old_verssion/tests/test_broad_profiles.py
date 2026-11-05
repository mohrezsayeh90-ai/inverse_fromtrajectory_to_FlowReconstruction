import numpy as np
from channel_operator.broad_profiles import sample_inlets


def test_paper_based_inputs_preserve_channel_constraints():
    bank = sample_inlets(24, 123)
    profiles, y = bank['profiles'], bank['y']
    assert profiles.shape == (24, 257)
    assert profiles.min() >= 0 and profiles.max() <= 2.875
    assert np.all(profiles[:, [0,-1]] == 0)
    assert np.max(abs(np.trapezoid(profiles, y, axis=1)-1)) < 1e-6
    assert np.all((bank['reynolds'] >= 10) & (bank['reynolds'] <= 100))
    assert len(np.unique(bank['length_scales'])) == 6
    assert 'fields' not in bank


def test_input_banks_are_reproducible_and_seeded_independently():
    first = sample_inlets(12, 8)
    repeat = sample_inlets(12, 8)
    heldout = sample_inlets(12, 9)
    np.testing.assert_array_equal(first['profiles'], repeat['profiles'])
    assert not np.any(np.all(first['profiles'][:,None] == heldout['profiles'][None], axis=-1))

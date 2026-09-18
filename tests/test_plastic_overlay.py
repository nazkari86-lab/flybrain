import numpy as np
import pytest

from flybrain.plastic_overlay import PlasticWeightOverlay


def test_overlay_starts_at_unit_multipliers_without_mutating_canonical_weights() -> None:
    canonical_weights = np.array([3.0, 5.0, 7.0], dtype=np.float32)
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([1, 2], dtype=np.int64),
        canonical_edge_count=3,
    )

    assert overlay.edge_indices.tolist() == [1, 2]
    assert overlay.multipliers.tolist() == [1.0, 1.0]
    assert overlay.apply_to(canonical_weights).tolist() == [3.0, 5.0, 7.0]
    assert canonical_weights.tolist() == [3.0, 5.0, 7.0]


def test_overlay_accepts_only_bounded_finite_multipliers_and_resets_exactly() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([0, 2], dtype=np.int64),
        canonical_edge_count=3,
    )

    overlay.set_multipliers(
        np.array([0.2, 1.5], dtype=np.float32), minimum=0.2, maximum=1.5
    )
    first_digest = overlay.digest()
    assert overlay.multipliers.tolist() == pytest.approx([0.2, 1.5])
    assert first_digest == overlay.digest()

    overlay.reset()
    assert overlay.multipliers.tolist() == [1.0, 1.0]
    assert overlay.digest() != first_digest

    with pytest.raises(ValueError, match="bounds"):
        overlay.set_multipliers(np.array([0.1, 1.0], dtype=np.float32), minimum=0.2, maximum=1.5)
    with pytest.raises(ValueError, match="finite"):
        overlay.set_multipliers(np.array([np.nan, 1.0], dtype=np.float32), minimum=0.2, maximum=1.5)
    with pytest.raises(ValueError, match="finite"):
        overlay.set_multipliers(
            np.array([1.0, 1.0], dtype=np.float32), minimum=0.2, maximum=float("inf")
        )


def test_overlay_copy_keeps_an_independent_sparse_state() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([0], dtype=np.int64), canonical_edge_count=2
    )
    overlay.set_multipliers(np.array([0.5], dtype=np.float32), minimum=0.2, maximum=1.0)

    copied = overlay.copy()
    overlay.reset()

    assert copied.multipliers.tolist() == [0.5]
    assert copied.edge_indices.tolist() == [0]


def test_overlay_serialization_restores_exact_sparse_state_without_aliasing() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([0, 3], dtype=np.int64), canonical_edge_count=4
    )
    overlay.set_multipliers(np.array([0.5, 0.7], dtype=np.float32), minimum=0.2, maximum=1.0)

    restored = PlasticWeightOverlay.deserialize(overlay.serialize())
    overlay.reset()

    assert restored.digest() != overlay.digest()
    assert restored.edge_indices.tolist() == [0, 3]
    assert restored.multipliers.tolist() == pytest.approx([0.5, 0.7])


def test_overlay_serialization_rejects_tampered_sparse_state() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([0], dtype=np.int64), canonical_edge_count=2
    )
    payload = overlay.serialize()
    payload["multipliers"] = [0.5]

    with pytest.raises(ValueError, match="digest"):
        PlasticWeightOverlay.deserialize(payload)

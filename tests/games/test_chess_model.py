import chess
import torch

from flybrain.games.chess_model import ChessPolicyValueNet, policy_value


def test_network_scores_only_legal_moves() -> None:
    board = chess.Board()
    model = ChessPolicyValueNet(seed=7)

    priors, value = policy_value(model, board, torch.device("cpu"))

    assert set(priors) == set(board.legal_moves)
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_same_seed_initializes_identical_model_and_gradients_are_finite() -> None:
    first = ChessPolicyValueNet(seed=11)
    second = ChessPolicyValueNet(seed=11)

    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    batch = torch.zeros((2, 21, 8, 8), dtype=torch.float32)
    from_logits, to_logits, promotion_logits, value = first(batch)
    loss = (
        from_logits.square().mean()
        + to_logits.square().mean()
        + promotion_logits.square().mean()
        + (value - 0.5).square().mean()
    )
    loss.backward()

    gradients = [parameter.grad for parameter in first.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)
    assert any(torch.any(gradient != 0) for gradient in gradients)

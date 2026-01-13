import torch
import pytest 
import matplotlib.pyplot as plt 
from src2.models.memory import NeuralMemory, TitanMemoryConfig, memory_step

def test_single_sample_convergence():
    """
    Verifying that a single memory module can learn a static pattern
    for a single sequence.
    """

    torch.manual_seed(42)

    config = TitanMemoryConfig(
        dim = 16,
        hidden_multiplier = 4,
        layers = 2,
        alpha = 0.0,
        learning_rate = 0.1,
    )

    model = NeuralMemory(config.dim, config.hidden_multiplier * config.dim, config.dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fact_key = torch.randn(config.dim)
    fact_value = torch.randn(config.dim)

    params = model.get_init_params(device)
    buffers = {}
    surprises = []
    print("Starting simple test training...")

    for step in range(20):
        params, pred, buffers, surprise = memory_step(  
            model,
            params,
            buffers,
            fact_key,
            fact_value,
            learning_rate=config.learning_rate,
            decay=config.alpha,
            mu=0.9
        )
        surprises.append(surprise.item())
        print(f"Step {step}: Surprise = {surprise.item(): .4f}")

    initial_surprise = surprises[0]
    final_surprise = surprises[-1]
    
    assert final_surprise < initial_surprise
    assert final_surprise < 0.1, f"Memory did not converge! final surprise: {final_surprise}"

    print(f"Test passed! final surprise: {final_surprise}")

def test_forgetting_mechanism():
    """

    """
    torch.manual_seed(42)
    model = NeuralMemory(16, 32, 16)

    params = model.get_init_params()
    buffers = {}

    key = torch.randn(16)
    value = torch.randn(16)

    w_norm_before = params["w1"].norm()

    decay_rate = 0.5
    params, _, _, _ = memory_step(
        model,
        params,
        buffers,
        key,
        value,
        learning_rate=0.0,
        decay=decay_rate,
        mu=0.0
    )

    w_norm_after = params["w1"].norm()
    assert w_norm_after < w_norm_before, "Forgetting mechanism failed!"
    assert torch.isclose(w_norm_after, w_norm_before * decay_rate, atol=1e-3), "Decay math is incorrect."

    print("Test passed! Forgetting mechanism works.")

if __name__ == "__main__":
    test_single_sample_convergence()
    test_forgetting_mechanism()

    
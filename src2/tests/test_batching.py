import torch

from src2.models.memory import NeuralMemory, TitanMemoryConfig, MemoryRNN

def test_batch_independence():
    torch.manual_seed(42)
    
    # 1. Config
    # Learning rate 0.01 is standard for SGD-based memory updates
    config = TitanMemoryConfig(dim=8, alpha=0.0, learning_rate=0.1)
    arch = NeuralMemory(8, 32, 8)
    model = MemoryRNN(config, arch)
    
    # 2. Data
    # Batch 0: Constant pattern (A -> A). Easiest thing to learn.
    # Batch 1: Random noise. Hard/Impossible to predict.
    steps = 50
    keys = torch.randn(2, steps, 8)
    values = torch.randn(2, steps, 8)
    
    # Enforce constant pattern for Batch 0
    keys[0, :, :] = keys[0, 0, :]
    values[0, :, :] = values[0, 0, :]
    
    # 3. Run
    preds, surprises = model(keys, values)
    
    # 4. Analysis
    final_surprise_constant = surprises[0, -1].item()
    final_surprise_random = surprises[1, -1].item()
    
    print(f"\nFinal Surprise (Constant): {final_surprise_constant:.5f}")
    print(f"Final Surprise (Random):   {final_surprise_random:.5f}")
    
    # 5. Assertions
    # Convergence: Constant pattern should have learned (error near 0)
    assert final_surprise_constant < 1.0, f"Memory failed to learn constant pattern (Got {final_surprise_constant})"
    
    # Independence: Random pattern should have much higher error
    # If they are equal, batching is broken (averaging).
    assert final_surprise_random > final_surprise_constant * 2.0, "Model did not distinguish between Easy and Hard patterns!"
    
    print("Test Passed: Memory works and batches are independent.")

if __name__ == "__main__":
    test_batch_independence()
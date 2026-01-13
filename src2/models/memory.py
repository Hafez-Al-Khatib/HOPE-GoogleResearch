from dataclasses import dataclass
from torch.func import grad_and_value, vmap
import torch 
import torch.nn as nn 
import torch.nn.functional as  F 

@dataclass
class TitanMemoryConfig:
    dim: int
    hidden_multiplier: int = 4
    layers: int = 2
    alpha: float = 0.02
    learning_rate: float = 1e-3


class NeuralMemory(nn.Module):
    """
    A Stateless MLP. Weights are not stored in self.parameters()
    during the functional pass. They are provided externally.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.w1 = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.w2 = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.act = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        return self.stateless_forward(x, {'w1':self.w1, 'w2':self.w2})
    
    def stateless_forward(self, x, params):
        w1 = params['w1']
        w2 = params['w2']
        x_norm = F.normalize(x, p=2, dim=-1)
        h = x_norm @ w1.transpose(-1, -2)
        h = self.act(h)
        out = h @ w2.transpose(-1, -2)
        return out

    def get_init_params(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        return {
            k: v.detach().clone().to(device) for k, v in self.named_parameters()
        }

def memory_step(model, params, buffers, key, value, learning_rate, decay, mu):
    """
    Performs one step of inference training for one sample only. 

    Args:
        model: The model to be trained.
        params: Dict of {name: tensor} representing current weights or memory states.
        key: Input Tensor [dim]
        value: Target Tensor [dim]
        learning_rate: float , step_size [eta]
        decay: float, decay or forgetting rate [alpha] (weight decay factor).
        mu: float, momentum factor [mu].

    Returns:
        new_params: updated weights.
        pred: the prediction made before the update.
        surprise: gradient norm as a scalar.
    """

    def compute_loss(p, k, v):
        pred = model.stateless_forward(k, p)
        loss = torch.sum((pred - v) ** 2)
        return loss, pred
    
    grads, (loss_val, pred) = grad_and_value(compute_loss, has_aux=True)(params, key, value)
    grad_flat = torch.cat([g.reshape(-1) for g in grads.values() if g is not None])
    grad_norm = torch.norm(grad_flat)
    surprise = grad_norm.detach()

    max_grad_norm = 1.0
    clip_coef = torch.clamp(max_grad_norm / (grad_norm + 1e-6), max=1.0)

    new_params = {}
    new_buffers = {}

    for name, param in params.items():
        if name in grads:
            g = grads[name]
            # Adding NaN guard
            g = torch.nan_to_num(g, nan=0.0)
            g = g * clip_coef
            buf_key = f"momentum_{name}"
            v_prev = buffers.get(buf_key, torch.zeros_like(param))
            v_new = mu * v_prev + g
            p_decayed = param * (1 - decay)
            p_new = p_decayed - (learning_rate * v_new)
            new_params[name] = p_new
            new_buffers[buf_key] = v_new
        else:
            new_params[name] = param
            # Carry forward old buffer
            buf_key = f"momentum_{name}"
            if buf_key in buffers:
                new_buffers[buf_key] = buffers[buf_key]

    return new_params, new_buffers, pred, surprise

class MemoryRNN(nn.Module):
    def __init__(self, config: TitanMemoryConfig, architecture: NeuralMemory):
        super().__init__()
        self.model = architecture
        self.config = config

    def forward(self, keys, values):
        batched_params, batched_buffers = self._init_batched_states(keys.shape[0], keys.device)
        all_surprises = []
        all_preds = []

        def step_wrapper(p, b, k, v):
            return memory_step(
                self.model,
                p, b, k, v,
                self.config.learning_rate,
                self.config.alpha,
                mu=0.9
            )
        batched_step = vmap(step_wrapper, in_dims=(0, 0, 0, 0))

        for t in range(keys.shape[1]):
            k_t = keys[:, t, :]
            v_t = values[:, t, :]
            batched_params, batched_buffers, pred, surprise = batched_step(
                batched_params,
                batched_buffers,
                k_t,
                v_t,
            )

            all_preds.append(pred)
            all_surprises.append(surprise)

        return torch.stack(all_preds, dim=1), torch.stack(all_surprises, dim=1)

    def _init_batched_states(self, batch_size, device):
        single_params = self.model.get_init_params(device)
        batched_params = {k: v.unsqueeze(0).expand(batch_size, *v.shape).contiguous() for k, v in single_params.items()}
        batched_buffers = {}
        for k, v in batched_params.items():
            batched_buffers[f"momentum_{k}"] = torch.zeros_like(v)
        return batched_params, batched_buffers
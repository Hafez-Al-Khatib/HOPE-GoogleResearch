import torch
import torch.nn as nn
import torch.nn.functional as F

class DGDUpdate(nn.Module):
    def forward(self, memory, k, v, alpha, eta):
        """
        Args:
            memory: Current fast weight matrix [B, H, Dim, Dim]
            k: input Key vector [B, H, Dim, 1]
            v: input Value vector [B, H, Dim, 1]
            alpha: Forgetting rate [B, H, 1, 1]
            eta: Learning rate [B, H, 1, 1]
        """

        memory_decayed = memory * alpha
        prediction = memory @ k

        error = prediction - v
        gradient = error @ k.transpose(-1, -2)
        
        memory_new = memory_decayed + eta * gradient
        return memory_new

# Self-Referential Titans (Working Memory)
class SRT(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.proj_in = nn.Linear(dim, 3*dim)

        # The controller is essentially a small network that decides how to learn.
        # It controls eta (This token is important) vs alpha (This token is noise).

        self.controller = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 5*num_heads), # Outputs: K, V, Q, alpha, eta values
        )

        self.proj_out = nn.Linear(dim, dim)
        self.dgd_update = DGDUpdate()
    
    def forward(self, x, memory_state=None):
        # x shape: [B, seq_len, dim]
        b, seq_len, _ = x.shape

        if memory_state is None:
            memory_state = torch.zeros((b, self.num_heads, self.head_dim, self.head_dim), device=x.device)
        
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            controls = self.controller(x_t).view(b, self.num_heads, 5)

            gate_k = controls[..., 0].unsqueeze(-1)
            gate_v = controls[..., 1].unsqueeze(-1)
            gate_q = controls[..., 2].unsqueeze(-1)
            eta = torch.sigmoid(controls[..., 3]).unsqueeze(-1).unsqueeze(-1)
            alpha = torch.sigmoid(controls[..., 4]).unsqueeze(-1).unsqueeze(-1)

            base_params = self.proj_in(x_t).view(b, self.num_heads, 3, self.head_dim)
            k_base, v_base, q_base = base_params.unbind(dim=2)

            k_t = (k_base * gate_k).unsqueeze(-1)
            v_t = (v_base * gate_v).unsqueeze(-1)
            q_t = (q_base * gate_q).unsqueeze(-1)
            
            y_t = (memory_state @ q_t).squeeze(-1)
            outputs.append(y_t)


            memory_state = self.dgd_update(memory_state, k_t, v_t, alpha, eta)
        
        outputs = torch.stack(outputs, dim=1).flatten(2)
        return self.proj_out(outputs), memory_state

# CMS Long Term Storage
# This replaces the MLP layer. It is a stack of distinct networks.
class ContinuumMemorySystem(nn.Module):
    def __init__(self, dim, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_levels)
        ])

    def forward(self, x):
        for mlp in self.levels:
            x = x + mlp(x)
        return x

class HOPELayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.titans = SRT(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim) 
        self.cms = ContinuumMemorySystem(dim)
    
    def forward(self, x, memory_state=None):
        titan_out, new_memory = self.titans(self.norm1(x), memory_state)
        x = x + titan_out
        x = self.norm2(x)
        x = self.cms(x)
        return x, new_memory


def simulate_clock_logic():
    print("\n--- SIMULATING CMS CLOCK LOGIC ---")
    print("Goal: Demonstrate how layers are frozen based on the fixed schedule.")
    
    dim = 64
    model = HOPELayer(dim=dim, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # The Fixed Schedule (The critique of HOPE argues this is too rigid)
    cms_frequencies = [1, 5, 10] 
    
    for step in range(1, 16): 
        # --- THE CLOCK (Gating Logic) ---
        active_levels = []
        for level_idx, freq in enumerate(cms_frequencies):
            
            # Clock-based update
            if step % freq == 0:
                active_levels.append(level_idx)
                for param in model.cms.levels[level_idx].parameters():
                    param.requires_grad = True
            else:
                for param in model.cms.levels[level_idx].parameters():
                    param.requires_grad = False
        
        # PROPOSED IMPROVEMENT (Liquid-HOPE):
        # In the future, we would replace 'if step % freq' with:
        # if input_surprise > threshold: unfreeze_slow_layer()

        # Dummy Step
        inputs = torch.randn(1, 10, dim)
        output, _ = model(inputs)
        loss = output.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        status = f"Step {step:02d}: Active Gears {active_levels}"
        if 2 in active_levels: status += " [SLOW GEAR UPDATE]"
        print(status)

if __name__ == "__main__":
    simulate_clock_logic()

        
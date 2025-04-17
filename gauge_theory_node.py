import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GaugeTheoryNode(nn.Module):
    def __init__(self, dimension=512, gauge_group="U(1)", lattice_size=10):
        super().__init__()
        self.dimension = dimension
        self.gauge_group = gauge_group
        self.lattice_size = lattice_size
        
        # Initialize gauge field with proper shape
        self.field = nn.Parameter(torch.randn(lattice_size, lattice_size, lattice_size, 4))
        
        # Field processors with correct dimensions
        self.field_processor = nn.Sequential(
            nn.Linear(4, dimension // 2),
            nn.ReLU(),
            nn.Linear(dimension // 2, dimension)
        )
        
        self.wilson_processor = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.ReLU(),
            nn.Linear(dimension // 2, dimension // 4),
            nn.ReLU(),
            nn.Linear(dimension // 4, 1)
        )
        
        self.topology_processor = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.ReLU(),
            nn.Linear(dimension // 2, 1)
        )
        
        # Wilson loop parameters
        self.wilson_loop_size = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, x):
        # Ensure input has correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != 4:
            x = x[:, :4]
            
        # Process input through field processor
        field_output = self.field_processor(x)
        
        # Compute Wilson loop
        wilson_loop = self.compute_wilson_loop(x)
        wilson_output = self.wilson_processor(field_output)
        
        # Compute topology
        topology_output = self.topology_processor(field_output)
        
        return {
            'field_output': field_output,
            'wilson_loop': wilson_output,
            'topology': topology_output
        }
    
    def compute_wilson_loop(self, x):
        # Extract coordinates and ensure they're within bounds
        coords = x[0, :4].long() % self.lattice_size
        
        # Initialize loop tensor
        loop = torch.ones(1, device=x.device)
        
        # Compute Wilson loop around plaquette
        for mu in range(4):
            for nu in range(mu + 1, 4):
                # Get links around plaquette with proper indexing
                pos = coords.clone()
                link1 = self.field[pos[0], pos[1], pos[2], mu]
                pos[mu] = (pos[mu] + 1) % self.lattice_size
                link2 = self.field[pos[0], pos[1], pos[2], nu]
                pos[nu] = (pos[nu] + 1) % self.lattice_size
                link3 = -self.field[pos[0], pos[1], pos[2], mu]  # Negative for reverse direction
                pos[mu] = pos[mu] - 1
                link4 = -self.field[coords[0], coords[1], coords[2], nu]  # Negative for reverse direction
                
                # Multiply links with proper orientation
                plaquette = torch.exp(link1 + link2 + link3 + link4)
                loop = loop * plaquette
        
        return loop
    
    def compute_action(self):
        action = torch.zeros(1, device=self.field.device)
        
        # Sum over all plaquettes
        for mu in range(4):
            for nu in range(mu + 1, 4):
                for i in range(self.lattice_size):
                    for j in range(self.lattice_size):
                        for k in range(self.lattice_size):
                            x = torch.tensor([[i, j, k, 0]], dtype=torch.float32, device=self.field.device)
                            plaquette = self.compute_wilson_loop(x)
                            action = action + (1 - plaquette.real)
        
        return action
    
    def update_field(self, learning_rate=0.01):
        # Compute action and update field
        action = self.compute_action()
        action.backward()
        
        with torch.no_grad():
            if self.field.grad is not None:
                self.field.data -= learning_rate * self.field.grad
                self.field.grad.zero_()
    
    def visualize_field(self, filename="gauge_field.png"):
        # Create 3D plot of field configuration
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract field magnitude for color mapping
        field_mag = torch.norm(self.field, dim=-1).detach().numpy()
        
        # Create coordinate grids
        x, y, z = np.meshgrid(
            np.arange(self.lattice_size),
            np.arange(self.lattice_size),
            np.arange(self.lattice_size)
        )
        
        # Plot field vectors with magnitude-based coloring
        quiver = ax.quiver(
            x, y, z,
            self.field[:, :, :, 0].detach().numpy(),
            self.field[:, :, :, 1].detach().numpy(),
            self.field[:, :, :, 2].detach().numpy(),
            length=0.5,
            normalize=True,
            cmap='viridis',
            array=field_mag.flatten()
        )
        
        # Add colorbar
        plt.colorbar(quiver)
        plt.savefig(filename)
        plt.close()
    
    def get_field_configuration(self):
        with torch.no_grad():
            wilson_loop = self.compute_wilson_loop(
                torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.field.device)
            )
            
            return {
                'lattice_size': self.lattice_size,
                'gauge_group': self.gauge_group,
                'field': self.field.detach().cpu().numpy(),
                'wilson_loop_size': wilson_loop.item()
            } 
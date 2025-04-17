import sympy as sp
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from calculus_engine import CalculusEngine
from knowledge_source import KnowledgeSource, KnowledgeEntry

class PhysicsError(Exception):
    """Custom exception for physics calculation errors."""
    pass

@dataclass
class PhysicsResult:
    value: Union[float, sp.Expr]
    units: str
    steps: List[str]
    plot_data: Optional[Dict] = None
    confidence: float = 1.0

class PhysicsEngine:
    def __init__(self):
        self.calculus = CalculusEngine()
        self.knowledge_source = KnowledgeSource()
        
        # Physical constants
        self.constants = {
            'G': 6.67430e-11,  # Gravitational constant (m^3 kg^-1 s^-2)
            'c': 299792458,    # Speed of light (m/s)
            'h': 6.62607015e-34,  # Planck constant (J s)
            'e': 1.602176634e-19,  # Elementary charge (C)
            'me': 9.1093837015e-31,  # Electron mass (kg)
            'mp': 1.67262192369e-27,  # Proton mass (kg)
            'k': 1.380649e-23,  # Boltzmann constant (J/K)
            'NA': 6.02214076e23,  # Avogadro's number (mol^-1)
            'R': 8.31446261815324,  # Gas constant (J mol^-1 K^-1)
            'epsilon0': 8.8541878128e-12,  # Vacuum permittivity (F/m)
            'mu0': 1.25663706212e-6,  # Vacuum permeability (H/m)
            'hbar': 1.054571817e-34,  # Reduced Planck constant (J s)
            'm_e': 9.1093837015e-31,  # Electron mass (kg)
            'k_B': 1.380649e-23,  # Boltzmann constant (J/K)
        }
        
        # Initialize knowledge base with physics concepts
        self._initialize_physics_knowledge()
    
    def _initialize_physics_knowledge(self):
        """Initialize the knowledge base with physics concepts"""
        physics_concepts = [
            KnowledgeEntry(
                source='physics/mechanics',
                content='Classical mechanics describes the motion of macroscopic objects under the influence of forces.',
                metadata={
                    'type': 'concept',
                    'category': 'mechanics',
                    'subcategory': 'classical',
                    'keywords': ['force', 'motion', 'velocity', 'acceleration', 'momentum']
                },
                confidence=1.0
            ),
            KnowledgeEntry(
                source='physics/electromagnetism',
                content='Electromagnetism studies the interaction between electric charges and currents.',
                metadata={
                    'type': 'concept',
                    'category': 'electromagnetism',
                    'subcategory': 'classical',
                    'keywords': ['electric', 'magnetic', 'field', 'charge', 'current']
                },
                confidence=1.0
            ),
            KnowledgeEntry(
                source='physics/quantum',
                content='Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic scales.',
                metadata={
                    'type': 'concept',
                    'category': 'quantum',
                    'subcategory': 'mechanics',
                    'keywords': ['quantum', 'wavefunction', 'uncertainty', 'superposition']
                },
                confidence=1.0
            ),
            KnowledgeEntry(
                source='physics/relativity',
                content='Relativity describes the relationship between space and time, and the effects of gravity.',
                metadata={
                    'type': 'concept',
                    'category': 'relativity',
                    'subcategory': 'general',
                    'keywords': ['relativity', 'spacetime', 'gravity', 'curvature']
                },
                confidence=1.0
            ),
            KnowledgeEntry(
                source='physics/thermodynamics',
                content='Thermodynamics studies the relationships between heat, work, and energy.',
                metadata={
                    'type': 'concept',
                    'category': 'thermodynamics',
                    'subcategory': 'classical',
                    'keywords': ['heat', 'work', 'energy', 'entropy', 'temperature']
                },
                confidence=1.0
            )
        ]
        
        for entry in physics_concepts:
            self.knowledge_source.update_knowledge_base(entry)
    
    def calculate_force(self, mass: float, acceleration: float) -> PhysicsResult:
        """Calculate force using Newton's second law: F = ma"""
        try:
            force = mass * acceleration
            steps = [
                f"Using Newton's second law: F = ma",
                f"F = {mass} kg * {acceleration} m/s²",
                f"F = {force} N"
            ]
            return PhysicsResult(
                value=force,
                units="N",
                steps=steps
            )
        except Exception as e:
            raise ValueError(f"Error calculating force: {str(e)}")
    
    def calculate_energy(self, mass: float, velocity: float = None, height: float = None) -> PhysicsResult:
        """Calculate kinetic or potential energy"""
        try:
            if velocity is not None:
                # Kinetic energy
                energy = 0.5 * mass * velocity**2
                steps = [
                    f"Using kinetic energy formula: KE = ½mv²",
                    f"KE = 0.5 * {mass} kg * ({velocity} m/s)²",
                    f"KE = {energy} J"
                ]
                return PhysicsResult(
                    value=energy,
                    units="J",
                    steps=steps
                )
            elif height is not None:
                # Potential energy
                energy = mass * self.constants['G'] * height
                steps = [
                    f"Using gravitational potential energy formula: PE = mgh",
                    f"PE = {mass} kg * {self.constants['G']} m³/kg/s² * {height} m",
                    f"PE = {energy} J"
                ]
                return PhysicsResult(
                    value=energy,
                    units="J",
                    steps=steps
                )
            else:
                raise ValueError("Either velocity or height must be provided")
        except Exception as e:
            raise ValueError(f"Error calculating energy: {str(e)}")
    
    def calculate_electric_field(self, charge: float, distance: float) -> PhysicsResult:
        """Calculate electric field due to a point charge"""
        try:
            field = (self.constants['k'] * charge) / (distance**2)
            steps = [
                f"Using Coulomb's law for electric field: E = kq/r²",
                f"E = ({self.constants['k']} N·m²/C² * {charge} C) / ({distance} m)²",
                f"E = {field} N/C"
            ]
            return PhysicsResult(
                value=field,
                units="N/C",
                steps=steps
            )
        except Exception as e:
            raise ValueError(f"Error calculating electric field: {str(e)}")
    
    def calculate_quantum_energy(self, frequency: float) -> PhysicsResult:
        """Calculate quantum energy using Planck's equation: E = hf"""
        try:
            energy = self.constants['h'] * frequency
            steps = [
                f"Using Planck's equation: E = hf",
                f"E = {self.constants['h']} J·s * {frequency} Hz",
                f"E = {energy} J"
            ]
            return PhysicsResult(
                value=energy,
                units="J",
                steps=steps
            )
        except Exception as e:
            raise ValueError(f"Error calculating quantum energy: {str(e)}")
    
    def calculate_relativistic_energy(self, mass: float, velocity: float) -> PhysicsResult:
        """Calculate relativistic energy using Einstein's equation"""
        try:
            gamma = 1 / sp.sqrt(1 - (velocity**2 / self.constants['c']**2))
            energy = gamma * mass * self.constants['c']**2
            steps = [
                f"Using relativistic energy equation: E = γmc²",
                f"γ = 1/√(1 - v²/c²) = {gamma}",
                f"E = {gamma} * {mass} kg * ({self.constants['c']} m/s)²",
                f"E = {energy} J"
            ]
            return PhysicsResult(
                value=energy,
                units="J",
                steps=steps
            )
        except Exception as e:
            raise ValueError(f"Error calculating relativistic energy: {str(e)}")
    
    def calculate_entropy(self, heat: float, temperature: float) -> PhysicsResult:
        """Calculate entropy change: ΔS = Q/T"""
        try:
            entropy = heat / temperature
            steps = [
                f"Using entropy formula: ΔS = Q/T",
                f"ΔS = {heat} J / {temperature} K",
                f"ΔS = {entropy} J/K"
            ]
            return PhysicsResult(
                value=entropy,
                units="J/K",
                steps=steps
            )
        except Exception as e:
            raise ValueError(f"Error calculating entropy: {str(e)}")
    
    def solve_wave_equation(self, amplitude: float, frequency: float, phase: float = 0) -> PhysicsResult:
        """Solve the wave equation for a given amplitude, frequency, and phase"""
        try:
            t = sp.Symbol('t')
            wave = amplitude * sp.sin(2 * sp.pi * frequency * t + phase)
            
            # Generate plot data
            t_vals = np.linspace(0, 1/frequency, 100)
            y_vals = [float(wave.subs(t, t_val)) for t_val in t_vals]
            
            steps = [
                f"Wave equation: y(t) = A sin(2πft + φ)",
                f"y(t) = {amplitude} * sin(2π * {frequency} * t + {phase})"
            ]
            
            return PhysicsResult(
                value=wave,
                units="m",
                steps=steps,
                plot_data={'t': t_vals.tolist(), 'y': y_vals}
            )
        except Exception as e:
            raise ValueError(f"Error solving wave equation: {str(e)}")
    
    def calculate_orbital_velocity(self, mass: float, radius: float) -> PhysicsResult:
        """Calculate orbital velocity for a circular orbit"""
        try:
            velocity = sp.sqrt(self.constants['G'] * mass / radius)
            steps = [
                f"Using orbital velocity formula: v = √(GM/r)",
                f"v = √({self.constants['G']} m³/kg/s² * {mass} kg / {radius} m)",
                f"v = {velocity} m/s"
            ]
            return PhysicsResult(
                value=velocity,
                units="m/s",
                steps=steps
            )
        except Exception as e:
            raise ValueError(f"Error calculating orbital velocity: {str(e)}")
    
    def calculate_quantum_probability(self, wavefunction: sp.Expr, position: float) -> PhysicsResult:
        """Calculate quantum probability density at a given position"""
        try:
            probability = abs(wavefunction.subs(sp.Symbol('x'), position))**2
            steps = [
                f"Using quantum probability formula: P(x) = |ψ(x)|²",
                f"P({position}) = |{wavefunction}|²",
                f"P({position}) = {probability}"
            ]
            return PhysicsResult(
                value=probability,
                units="",
                steps=steps
            )
        except Exception as e:
            raise ValueError(f"Error calculating quantum probability: {str(e)}")
    
    def calculate_blackbody_radiation(self, temperature: float, wavelength: float) -> PhysicsResult:
        """Calculate blackbody radiation intensity using Planck's law"""
        try:
            h = self.constants['h']
            c = self.constants['c']
            k = self.constants['k']
            
            intensity = (2 * h * c**2 / wavelength**5) / (sp.exp(h * c / (wavelength * k * temperature)) - 1)
            
            steps = [
                f"Using Planck's law for blackbody radiation",
                f"I(λ,T) = (2hc²/λ⁵) / (e^(hc/λkT) - 1)",
                f"I({wavelength} m, {temperature} K) = {intensity} W/m³"
            ]
            
            return PhysicsResult(
                value=intensity,
                units="W/m³",
                steps=steps
            )
        except Exception as e:
            raise ValueError(f"Error calculating blackbody radiation: {str(e)}")

    def solve_schrodinger_equation(self, potential: Callable, x_range: Tuple[float, float], n_points: int = 100) -> PhysicsResult:
        """Solve the time-independent Schrödinger equation for a given potential."""
        try:
            x = np.linspace(x_range[0], x_range[1], n_points)
            dx = x[1] - x[0]
            
            # Construct Hamiltonian matrix
            T = np.diag(-2 * np.ones(n_points)) + np.diag(np.ones(n_points-1), 1) + np.diag(np.ones(n_points-1), -1)
            T = -self.constants['hbar']**2 / (2 * self.constants['m_e'] * dx**2) * T
            
            V = np.diag([potential(xi) for xi in x])
            H = T + V
            
            # Solve for eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            
            # Normalize eigenvectors
            eigenvectors = eigenvectors / np.sqrt(dx)
            
            return PhysicsResult(
                value=(eigenvalues, eigenvectors),
                units=('eV', '1/√m'),
                steps=[
                    "Constructed Hamiltonian matrix",
                    "Solved eigenvalue problem",
                    "Normalized wavefunctions"
                ],
                plot_data={
                    'x': x,
                    'potential': [potential(xi) for xi in x],
                    'eigenvalues': eigenvalues[:5],  # First 5 energy levels
                    'wavefunctions': eigenvectors[:, :5]  # First 5 wavefunctions
                },
                confidence=0.95
            )
        except Exception as e:
            raise PhysicsError(f"Error solving Schrödinger equation: {str(e)}")

    def calculate_lorentz_transformation(self, velocity: float, position: Tuple[float, float, float], time: float) -> PhysicsResult:
        """Calculate Lorentz transformation for given velocity and spacetime coordinates."""
        try:
            gamma = 1 / np.sqrt(1 - (velocity / self.constants['c'])**2)
            
            # Transform coordinates
            t_prime = gamma * (time - velocity * position[0] / self.constants['c']**2)
            x_prime = gamma * (position[0] - velocity * time)
            y_prime = position[1]
            z_prime = position[2]
            
            return PhysicsResult(
                value=(t_prime, x_prime, y_prime, z_prime),
                units=('s', 'm', 'm', 'm'),
                steps=[
                    f"Calculated Lorentz factor: γ = {gamma}",
                    "Applied Lorentz transformation equations",
                    f"Transformed coordinates: t' = {t_prime:.3f} s, x' = {x_prime:.3f} m"
                ],
                confidence=0.98
            )
        except Exception as e:
            raise PhysicsError(f"Error calculating Lorentz transformation: {str(e)}")

    def calculate_boltzmann_distribution(self, temperature: float, energy_levels: List[float]) -> PhysicsResult:
        """Calculate Boltzmann distribution for given temperature and energy levels."""
        try:
            beta = 1 / (self.constants['k_B'] * temperature)
            partition_function = np.sum(np.exp(-beta * np.array(energy_levels)))
            probabilities = np.exp(-beta * np.array(energy_levels)) / partition_function
            
            return PhysicsResult(
                value=probabilities,
                units='dimensionless',
                steps=[
                    f"Calculated β = 1/(k_B*T) = {beta:.3e} J⁻¹",
                    f"Calculated partition function Z = {partition_function:.3f}",
                    "Computed Boltzmann probabilities"
                ],
                plot_data={
                    'energy_levels': energy_levels,
                    'probabilities': probabilities
                },
                confidence=0.97
            )
        except Exception as e:
            raise PhysicsError(f"Error calculating Boltzmann distribution: {str(e)}")

    def calculate_path_integral(self, action: Callable, time_range: Tuple[float, float], n_points: int = 100) -> PhysicsResult:
        """Calculate Feynman path integral for quantum systems."""
        try:
            t = np.linspace(time_range[0], time_range[1], n_points)
            dt = t[1] - t[0]
            
            # Calculate classical path
            classical_action = np.array([action(ti) for ti in t])
            
            # Compute quantum corrections
            quantum_phase = np.exp(1j * classical_action / self.constants['hbar'])
            propagator = np.sum(quantum_phase) * dt
            
            return PhysicsResult(
                value=propagator,
                units='dimensionless',
                steps=[
                    "Discretized time path",
                    "Calculated classical action",
                    "Computed quantum phase factor",
                    "Integrated over paths"
                ],
                plot_data={
                    't': t.tolist(),
                    'action': classical_action.tolist(),
                    'phase': np.angle(quantum_phase).tolist()
                },
                confidence=0.9
            )
        except Exception as e:
            raise PhysicsError(f"Error calculating path integral: {str(e)}")

    def calculate_metric_tensor(self, mass: float, radius: float, coordinates: Tuple[float, float, float, float]) -> PhysicsResult:
        """Calculate Schwarzschild metric tensor components for general relativity."""
        try:
            t, r, theta, phi = coordinates
            rs = 2 * self.constants['G'] * mass / (self.constants['c']**2)  # Schwarzschild radius
            
            # Calculate metric components
            g_tt = -(1 - rs/r)
            g_rr = 1/(1 - rs/r)
            g_theta = r**2
            g_phi = r**2 * np.sin(theta)**2
            
            metric = np.diag([g_tt, g_rr, g_theta, g_phi])
            
            return PhysicsResult(
                value=metric,
                units='m²',
                steps=[
                    f"Calculated Schwarzschild radius: rs = {rs:.3e} m",
                    "Computed metric tensor components",
                    f"g_tt = {g_tt:.3f}, g_rr = {g_rr:.3f}",
                    f"g_θθ = {g_theta:.3f}, g_φφ = {g_phi:.3f}"
                ],
                plot_data={
                    'coordinates': [t, r, theta, phi],
                    'metric_components': metric.tolist()
                },
                confidence=0.95
            )
        except Exception as e:
            raise PhysicsError(f"Error calculating metric tensor: {str(e)}")

    def calculate_partition_function_canonical(self, hamiltonian: Callable, temperature: float, states: List[Tuple]) -> PhysicsResult:
        """Calculate canonical partition function and thermodynamic quantities."""
        try:
            beta = 1 / (self.constants['k_B'] * temperature)
            
            # Calculate energies for each state
            energies = np.array([hamiltonian(state) for state in states])
            
            # Compute partition function
            Z = np.sum(np.exp(-beta * energies))
            
            # Calculate thermodynamic quantities
            F = -self.constants['k_B'] * temperature * np.log(Z)  # Free energy
            U = np.sum(energies * np.exp(-beta * energies)) / Z  # Internal energy
            S = (U - F) / temperature  # Entropy
            C = beta**2 * (np.sum(energies**2 * np.exp(-beta * energies)) / Z - (U**2))  # Heat capacity
            
            return PhysicsResult(
                value={
                    'partition_function': Z,
                    'free_energy': F,
                    'internal_energy': U,
                    'entropy': S,
                    'heat_capacity': C
                },
                units={
                    'partition_function': 'dimensionless',
                    'free_energy': 'J',
                    'internal_energy': 'J',
                    'entropy': 'J/K',
                    'heat_capacity': 'J/K'
                },
                steps=[
                    f"Calculated partition function Z = {Z:.3e}",
                    f"Free energy F = {F:.3e} J",
                    f"Internal energy U = {U:.3e} J",
                    f"Entropy S = {S:.3e} J/K",
                    f"Heat capacity C = {C:.3e} J/K"
                ],
                plot_data={
                    'energies': energies.tolist(),
                    'probabilities': np.exp(-beta * energies).tolist()
                },
                confidence=0.93
            )
        except Exception as e:
            raise PhysicsError(f"Error calculating partition function: {str(e)}")

    def calculate_quantum_field_propagator(self, mass: float, momentum: Tuple[float, float, float, float]) -> PhysicsResult:
        """Calculate quantum field theory propagator in momentum space."""
        try:
            p0, p1, p2, p3 = momentum
            p_squared = p0**2 - p1**2 - p2**2 - p3**2
            
            # Calculate Klein-Gordon propagator
            propagator = 1 / (p_squared - mass**2 + 1e-10j)
            
            # Calculate pole structure
            pole = np.sqrt(mass**2 + p1**2 + p2**2 + p3**2)
            
            return PhysicsResult(
                value=propagator,
                units='GeV⁻²',
                steps=[
                    f"Calculated 4-momentum squared: p² = {p_squared:.3e} GeV²",
                    f"Computed propagator: D(p) = 1/(p² - m²)",
                    f"Found pole at E = ±{pole:.3e} GeV"
                ],
                plot_data={
                    'momentum': [p0, p1, p2, p3],
                    'propagator_real': float(propagator.real),
                    'propagator_imag': float(propagator.imag)
                },
                confidence=0.92
            )
        except Exception as e:
            raise PhysicsError(f"Error calculating quantum field propagator: {str(e)}")

    def calculate_feynman_diagram(self, process: str, coupling: float, momenta: List[Tuple[float, float, float, float]]) -> PhysicsResult:
        """Calculate scattering amplitude from Feynman diagrams."""
        try:
            # Parse process string (e.g., "e+e- -> gamma -> mu+mu-")
            process_steps = process.split(" -> ")
            initial_state = process_steps[0]
            intermediate_state = process_steps[1] if len(process_steps) > 2 else None
            final_state = process_steps[-1]
            
            # Calculate vertex factors
            vertex_factor = coupling * np.sqrt(4 * np.pi)
            
            # Calculate propagators for internal lines
            propagators = []
            for p in momenta[2:-2]:  # Only internal lines
                p_squared = p[0]**2 - sum(pi**2 for pi in p[1:])
                # Add small imaginary part to avoid poles
                propagators.append(1 / (p_squared + 1e-10j))
            
            # Compute total amplitude
            amplitude = vertex_factor**2 * np.prod(propagators) if propagators else vertex_factor
            
            # Calculate cross-section
            phase_space = self._calculate_phase_space(momenta)
            cross_section = abs(amplitude)**2 * phase_space
            
            return PhysicsResult(
                value={'amplitude': amplitude, 'cross_section': cross_section},
                units={'amplitude': 'GeV⁻²', 'cross_section': 'pb'},
                steps=[
                    f"Parsed process: {initial_state} -> {intermediate_state + ' -> ' if intermediate_state else ''}{final_state}",
                    f"Calculated vertex factors: {vertex_factor:.3e}",
                    f"Number of propagators: {len(propagators)}",
                    f"Final amplitude: {amplitude:.3e}",
                    f"Cross-section: {cross_section:.3e} pb"
                ],
                plot_data={
                    'momenta': momenta,
                    'propagators': propagators
                },
                confidence=0.94
            )
        except Exception as e:
            raise PhysicsError(f"Error calculating Feynman diagram: {str(e)}")

    def simulate_binary_black_hole_merger(self, masses: Tuple[float, float], initial_separation: float, 
                                        time_steps: int = 1000) -> PhysicsResult:
        """Simulate binary black hole merger using numerical relativity."""
        try:
            m1, m2 = masses
            total_mass = m1 + m2
            reduced_mass = m1 * m2 / total_mass
            
            # Initialize arrays for simulation
            time = np.linspace(0, 1000, time_steps)
            separation = np.zeros(time_steps)
            gravitational_wave = np.zeros(time_steps, dtype=complex)
            
            # Initial conditions
            separation[0] = initial_separation
            orbital_frequency = np.sqrt(self.constants['G'] * total_mass / initial_separation**3)
            
            # Time evolution using post-Newtonian approximation
            for i in range(1, time_steps):
                # Update separation due to gravitational radiation
                energy_loss = -(32/5) * self.constants['G']**4 * reduced_mass**2 * total_mass**3 / (separation[i-1]**5 * self.constants['c']**5)
                separation[i] = max(separation[i-1] + energy_loss * (time[i] - time[i-1]), 2 * self.constants['G'] * total_mass / self.constants['c']**2)  # Minimum separation is Schwarzschild radius
                
                # Calculate gravitational waveform
                phase = 2 * orbital_frequency * time[i]
                amplitude = 4 * self.constants['G']**2 * reduced_mass * total_mass / (self.constants['c']**4 * separation[i])
                gravitational_wave[i] = amplitude * np.exp(1j * phase)
            
            return PhysicsResult(
                value={'waveform': gravitational_wave, 'separation': separation},
                units={'waveform': 'strain', 'separation': 'm'},
                steps=[
                    f"Initialized binary system: M = {total_mass:.2e} kg, μ = {reduced_mass:.2e} kg",
                    "Applied post-Newtonian approximation",
                    "Calculated gravitational waveform",
                    f"Final separation: {separation[-1]:.2e} m"
                ],
                plot_data={
                    'time': time.tolist(),
                    'separation': separation.tolist(),
                    'strain_real': gravitational_wave.real.tolist(),
                    'strain_imag': gravitational_wave.imag.tolist()
                },
                confidence=0.91
            )
        except Exception as e:
            raise PhysicsError(f"Error simulating black hole merger: {str(e)}")

    def solve_quantum_many_body(self, hamiltonian: np.ndarray, num_particles: int, 
                              temperature: float = 0) -> PhysicsResult:
        """Solve quantum many-body system using exact diagonalization."""
        try:
            # Construct many-body basis
            dimension = hamiltonian.shape[0]
            basis_states = self._generate_fock_basis(num_particles, dimension)
            
            # Project Hamiltonian onto many-body basis
            H_mb = self._project_hamiltonian(hamiltonian, basis_states)
            
            # Solve for eigenvalues and eigenvectors
            energies, states = np.linalg.eigh(H_mb)
            
            # Calculate observables
            density_matrix = self._calculate_density_matrix(states, energies, temperature)
            
            # Calculate one-body density matrix directly
            one_body_dm = np.zeros((dimension, dimension), dtype=complex)
            for i in range(dimension):
                for j in range(dimension):
                    # Calculate expectation value of c_i^dagger c_j
                    expectation = 0
                    for state in basis_states:
                        if i in state and j in state:
                            expectation += 1
                    one_body_dm[i,j] = expectation / len(basis_states)
            
            # Calculate correlation functions
            correlations = self._calculate_correlations(one_body_dm)
            
            return PhysicsResult(
                value={
                    'energies': energies,
                    'states': states,
                    'density_matrix': density_matrix,
                    'correlations': correlations
                },
                units='eV',
                steps=[
                    f"Constructed {len(basis_states)}-particle basis",
                    "Projected Hamiltonian onto many-body basis",
                    "Solved for eigenvalues and eigenvectors",
                    "Calculated density matrix and correlations"
                ],
                plot_data={
                    'energy_spectrum': energies.tolist(),
                    'correlation_matrix': correlations.tolist()
                },
                confidence=0.89
            )
        except Exception as e:
            raise PhysicsError(f"Error solving quantum many-body system: {str(e)}")

    def simulate_non_equilibrium_dynamics(self, initial_state: np.ndarray, liouvillian: Callable,
                                        time_range: Tuple[float, float], n_steps: int = 1000) -> PhysicsResult:
        """Simulate non-equilibrium quantum dynamics using Lindblad master equation."""
        try:
            time = np.linspace(time_range[0], time_range[1], n_steps)
            dt = time[1] - time[0]
            
            # Initialize arrays for time evolution
            state_history = np.zeros((n_steps,) + initial_state.shape, dtype=complex)
            state_history[0] = initial_state
            
            # Time evolution using Runge-Kutta 4th order
            for i in range(1, n_steps):
                k1 = liouvillian(state_history[i-1], time[i-1])
                k2 = liouvillian(state_history[i-1] + dt*k1/2, time[i-1] + dt/2)
                k3 = liouvillian(state_history[i-1] + dt*k2/2, time[i-1] + dt/2)
                k4 = liouvillian(state_history[i-1] + dt*k3, time[i-1] + dt)
                
                state_history[i] = state_history[i-1] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
            
            # Calculate observables
            observables = self._calculate_observables(state_history)
            entropy = self._calculate_von_neumann_entropy(state_history)
            
            return PhysicsResult(
                value={
                    'state_history': state_history,
                    'observables': observables,
                    'entropy': entropy
                },
                units='dimensionless',
                steps=[
                    "Initialized quantum system",
                    "Evolved using Lindblad master equation",
                    "Calculated time-dependent observables",
                    "Computed von Neumann entropy"
                ],
                plot_data={
                    'time': time.tolist(),
                    'observables': observables.tolist(),
                    'entropy': entropy.tolist()
                },
                confidence=0.92
            )
        except Exception as e:
            raise PhysicsError(f"Error simulating non-equilibrium dynamics: {str(e)}")

    def _calculate_phase_space(self, momenta: List[Tuple[float, float, float, float]]) -> float:
        """Helper function to calculate phase space factors."""
        try:
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            return (2 * np.pi)**4 * np.prod([1/(2*(p[0] + epsilon)) for p in momenta])
        except Exception as e:
            raise PhysicsError(f"Error calculating phase space: {str(e)}")

    def _generate_fock_basis(self, num_particles: int, dimension: int) -> List[np.ndarray]:
        """Generate many-body Fock basis states."""
        from itertools import combinations
        return [np.array(state) for state in combinations(range(dimension), num_particles)]

    def _project_hamiltonian(self, hamiltonian: np.ndarray, basis_states: List[np.ndarray]) -> np.ndarray:
        """Project single-particle Hamiltonian onto many-body basis."""
        dim = len(basis_states)
        H_mb = np.zeros((dim, dim), dtype=complex)
        for i, state_i in enumerate(basis_states):
            for j, state_j in enumerate(basis_states):
                H_mb[i,j] = np.sum(hamiltonian[state_i][:,state_j])
        return H_mb

    def _calculate_density_matrix(self, states: np.ndarray, energies: np.ndarray, 
                                temperature: float) -> np.ndarray:
        """Calculate density matrix at given temperature."""
        if temperature == 0:
            return np.outer(states[:,0], states[:,0].conj())
        else:
            beta = 1 / (self.constants['k_B'] * temperature)
            Z = np.sum(np.exp(-beta * energies))
            rho = np.zeros_like(states, dtype=complex)
            for i in range(len(energies)):
                rho += np.exp(-beta * energies[i]) * np.outer(states[:,i], states[:,i].conj())
            return rho / Z

    def _calculate_correlations(self, one_body_dm: np.ndarray) -> np.ndarray:
        """Calculate correlation functions from one-body density matrix."""
        return np.abs(one_body_dm)**2

    def _calculate_observables(self, state_history: np.ndarray) -> np.ndarray:
        """Calculate time-dependent observables from quantum state history."""
        return np.array([np.trace(state @ state.conj().T) for state in state_history])

    def _calculate_von_neumann_entropy(self, state_history: np.ndarray) -> np.ndarray:
        """Calculate von Neumann entropy from quantum state history."""
        entropy = np.zeros(len(state_history))
        for i, state in enumerate(state_history):
            eigenvalues = np.linalg.eigvalsh(state)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zero eigenvalues
            entropy[i] = -np.sum(eigenvalues * np.log(eigenvalues))
        return entropy

def main():
    # Initialize physics engine
    physics = PhysicsEngine()
    
    # Test QFT calculations with non-zero momenta
    print("\nTesting Feynman diagram calculation:")
    result = physics.calculate_feynman_diagram(
        "e+e- -> gamma -> mu+mu-",
        coupling=0.303,  # QED fine structure constant
        momenta=[(100,0,0,100), (-100,0,0,100), (1,0,0,200), (100,0,0,-100), (-100,0,0,-100)]  # Changed zero momentum to 1
    )
    print(f"Cross-section: {result.value['cross_section']:.2e} pb")
    
    # Test black hole merger with more realistic parameters
    print("\nTesting binary black hole merger:")
    result = physics.simulate_binary_black_hole_merger(
        masses=(30 * 2e30, 25 * 2e30),  # 30 and 25 solar masses
        initial_separation=1e9  # 1 million km
    )
    print(f"Final separation: {result.value['separation'][-1]:.2e} m")
    
    # Test quantum many-body calculation with proper dimensions
    print("\nTesting quantum many-body calculation:")
    H = np.random.random((4,4)) + 1j * np.random.random((4,4))
    H = H + H.conj().T  # Make Hermitian
    result = physics.solve_quantum_many_body(H, num_particles=2)
    print(f"Ground state energy: {result.value['energies'][0]:.3f} eV")
    print(f"Correlation matrix shape: {result.value['correlations'].shape}")
    
    # Test non-equilibrium dynamics
    print("\nTesting non-equilibrium dynamics:")
    initial_state = np.array([[1,0],[0,0]], dtype=complex)
    def liouvillian(rho, t):
        H = np.array([[0,1],[1,0]])
        return -1j * (H @ rho - rho @ H)
    result = physics.simulate_non_equilibrium_dynamics(initial_state, liouvillian, (0, 10))
    print(f"Final entropy: {result.value['entropy'][-1]:.3f}")

if __name__ == "__main__":
    main() 
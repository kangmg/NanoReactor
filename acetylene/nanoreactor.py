import numpy as np
import torch
import torchani
import math
from ase import Atoms
from ase.io import read
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
import random
import plotly.graph_objects as go
from os.path import splitext
import time
import ase
import warnings
from IPython.display import clear_output


random_seed = 0

def animodel(modelname:str):
    # check gpu
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if modelname == 'ANI1xnr':
        return torchani.models.ANI1xnr().to(device).ase()
    elif modelname == 'ANI1ccx':
        return torchani.models.ANI1ccx().to(device).ase()
    elif modelname == 'ANI2x':
        return torchani.models.ANI2x().to(device).ase()
    elif modelname == 'ANI1x':
        return torchani.models.ANI1x().to(device).ase()
    else:
        raise ValueError(f"Model {modelname} Not Found")


def random_vector():
    """Generate a random 3D unit vector."""
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])

def generate_acetylene_molecules(box_size, num_molecules, l_CH=1.06274817, l_CC=1.19506454, buffer_spacing=1.5, max_trials='auto'):
    """Generate multiple HCCH molecules in a box, with a grid-based placement using numpy."""
    ANG2NM = 1 / 10
    l_CH *= ANG2NM
    l_CC *= ANG2NM
    buffer_spacing *= ANG2NM
    
    # Calculate the grid size and grid count
    grid_size = (l_CC + 2 * l_CH) + buffer_spacing
    grid_count = math.ceil(box_size / grid_size) ** 3
    # Check if enough grids are available
    if num_molecules > grid_count:
        raise ValueError(f"Not enough grids. {num_molecules} molecules requested, but only {grid_count} grids available.")
    
    xyz = []  # List to store coordinates of all atoms
    elements = []  # List to store elements ('C', 'H')
    trials = 0
    used_grids = set()  # To keep track of which grids have been used

    # Generate molecules
    max_trials = 10 * num_molecules if max_trials == 'auto' else max_trials
    while len(xyz) // 4 < num_molecules and trials < max_trials:
        # Step 1: Randomly choose a grid
        available_grids = list(set(range(grid_count)) - used_grids)
        if not available_grids:
            raise RuntimeError("Ran out of available grids.")
        
        grid_index = np.random.choice(available_grids)
        used_grids.add(grid_index)

        # Step 2: Calculate grid coordinates
        grid_x = (grid_index % (box_size // grid_size)) * grid_size
        grid_y = ((grid_index // (box_size // grid_size)) % (box_size // grid_size)) * grid_size
        grid_z = (grid_index // ((box_size // grid_size) ** 2)) * grid_size

        # Step 3: Generate a random direction (unit vector) for the linear HCCH molecule
        direction = random_vector()

        # Calculate C1 and C2 positions (C1 and C2 are placed at l_CC distance apart)
        C1 = np.array([grid_x, grid_y, grid_z])
        C2 = C1 + direction * l_CC

        # Step 4: Calculate H1 and H2 positions (place H1 and H2 along the direction of the unit vector)
        H1 = C1 - direction * l_CH
        H2 = C2 + direction * l_CH

        # Step 5: Calculate centroid and shift all atoms to center the molecule
        centroid = (C1 + C2 + H1 + H2) / 4
        move_vector = np.array([grid_x, grid_y, grid_z]) - centroid
        
        # Apply the shift to all atoms
        C1 += move_vector
        C2 += move_vector
        H1 += move_vector
        H2 += move_vector

        # Step 6: Ensure the entire molecule stays within box bounds
        atoms = [C1, C2, H1, H2]
        
        # Check if the molecule stays within the box (all atoms must be inside)
        if all(np.all((0 <= atom) & (atom <= box_size)) for atom in atoms):
            # Add the positions and corresponding elements to the lists
            xyz.extend([C1, C2, H1, H2])
            elements.extend(['C', 'C', 'H', 'H'])
        else:
            # If the molecule goes out of bounds, do not mark the grid as used
            used_grids.remove(grid_index)

        trials += 1
    
    # If max_trials exceeded, raise error
    if len(xyz) // 4 < num_molecules:
        raise RuntimeError(f"Max trials exceeded. Only {len(xyz) // 4} molecules were generated.")
    
    return np.array(xyz), elements

get_box_length_nm = lambda concentration, num_molecules: ((num_molecules * M_ac) / (concentration * Na * 100))**(1/3) # in nm

ANG2NM = 1 / 10
Na = 6.022140857 # avogadro number pre-exponent
M_ac = 26.0 # acetylene molar weight

def save_to_xyz(xyz, elements, filename="system.xyz", unit_conversion=10):
    """Save the xyz coordinates and elements to an xyz file with unit conversion."""
    # Convert coordinates from nm to Å by multiplying by 10
    xyz *= unit_conversion

    # Calculate number of atoms
    num_atoms = len(xyz)

    # Open file for writing
    with open(filename, 'w') as f:
        # Write the number of atoms as the first line
        f.write(f"{num_atoms}\n")
        f.write("\n")
        # Write the coordinates and element types for each atom
        for symbol, (x, y, z) in zip(elements, xyz):
            f.write(f"{symbol}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")


# def visualize_xyz(atom, box_size_nm=False, group=None, save_name=None):
#     positions = atom.get_positions()
#     if group:
#         positions = positions[group]
#     xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
# 
#     if box_size_nm:
# 
#         # Box info
#         n = box_size_nm*10
#         x_coords = [0, n, n, 0, 0, n, n, 0]
#         y_coords = [0, 0, n, n, 0, 0, n, n]
#         z_coords = [0, 0, 0, 0, n, n, n, n]
# 
#         edges = [
#             (0, 1), (1, 2), (2, 3), (3, 0),  # 아래 네 모서리
#             (4, 5), (5, 6), (6, 7), (7, 4),  # 위 네 모서리
#             (0, 4), (1, 5), (2, 6), (3, 7)   # 위아래 연결 모서리
#         ]
# 
#         x_lines = []
#         y_lines = []
#         z_lines = []
# 
#         for edge in edges:
#             for vertex in edge:
#                 x_lines.append(x_coords[vertex])
#                 y_lines.append(y_coords[vertex])
#                 z_lines.append(z_coords[vertex])
#             x_lines.append(None)
#             y_lines.append(None)
#             z_lines.append(None)
# 
#     plots = data=[
#         # atoms
#         go.Scatter3d(
#             x=xs,
#             y=ys,
#             z=zs,
#             mode='markers',
#             marker=dict(size=1, color='gray')
#         ),
#         # scaling bar
#         go.Scatter3d(
#             x=[0, 10],
#             y=[0, 0],
#             z=[0, 0],
#             mode='lines+text',
#             line=dict(color='red', width=10),
#             text=['1 nm'],
#             textposition='top center',
#             textfont=dict(size=12, color='red'),
#             showlegend=False
#         )]
#     if box_size_nm:
#         plots.append(
#             go.Scatter3d(
#                     x=x_lines,
#                     y=y_lines,
#                     z=z_lines,
#                     mode='lines',
#                     line=dict(color='grey', width=3),
#                     showlegend=False
#                     )
#         )
#     # plot
#     fig = go.Figure(
#         plots
#     )
# 
#     # update layout
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(showgrid=False, zeroline=False, showline=False, title='', visible=False),
#             yaxis=dict(showgrid=False, zeroline=False, showline=False, title='', visible=False),
#             zaxis=dict(showgrid=False, zeroline=False, showline=False, title='', visible=False),
#             bgcolor='black'
#         ),
#         title=f'Num. atoms {len(atom)}',
#         paper_bgcolor='black'
#     )
# 
#     fig.update_scenes(
#         xaxis_showgrid=False,
#         yaxis_showgrid=False,
#         zaxis_showgrid=False,
#         xaxis_zeroline=False,
#         yaxis_zeroline=False,
#         zaxis_zeroline=False,
#     )
# 
#     fig.show()
#     if save_name:
#         fig.write_html(f'{save_name}.html')


def initial_system_builder(**params):
    """
    """
    # params
    conc = params.get('concentration')
    assert conc, "concentration is not given"
    buffer_spacing_Ang = params.get('buffer_spacing_Ang', 1.5)
    num_molecules = params.get('num_molecules')
    l_CC = params.get('l_CC', 1.19506454)
    l_CH = params.get('l_CH', 1.06274817)
    max_trials = params.get('max_trials', 10000)
    save = params.get('save', False)
    #display_system = params.get('display_system', False)
    filename = params.get('filename', 'auto')

    # reset filename
    if filename == 'auto':
        filename = f'box_{conc}[gcc-1]_molecules_{num_molecules}[molecule].xyz'

    #box size
    box_size_nm = get_box_length_nm(concentration=conc, num_molecules=num_molecules) # in nm

    xyz, elements = generate_acetylene_molecules(
        box_size=box_size_nm, 
        buffer_spacing=buffer_spacing_Ang,
        num_molecules=num_molecules, 
        max_trials=max_trials,
        l_CC=l_CC,
        l_CH=l_CH
        ) # in nm scale
    
    if save:
        save_to_xyz(
            xyz=xyz,
            elements=elements, 
            filename=filename,
            unit_conversion=10
            )
#    if display_system:
#        system = read(filename)
#        visualize_xyz(system, box_size_nm=box_size_nm)


class Parameters:
    """
    A simple class to manage parameters, allowing dictionary keys to be accessed like attributes.

    Example
    -------
    >>> params = Parameters({
            "concentration": 0.5,
            "temperature": 2500,
            "modeltype": 'ANI1xnr'
        })
    >>> print(params.temperature)
    2500
    >>> print(params.modeltype)
    'ANI1xnr'
    >>> params.temperature = 5000
    >>> print(params)
    {'concentration': 0.5, 'modeltype': 'ANI1xnr', 'temperature': 5000}
    """
    def __init__(self, parameters):
        self._parameters = parameters

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        raise AttributeError(f"parameters has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_parameters":
            super().__setattr__(name, value)
        else:
            self._parameters[name] = value

    def __repr__(self):
        from pprint import pformat
        return pformat(self._parameters, indent=0)




class NanoReactor:
    """
    NanoReactor simulation box

    Attributes
    ----------
    parameters : Parameters


    Methods
    -------
    run_simulation

    """
    # default parameters
    default_parameters = {
        'temperature_K': 2500,
        'time_step_fs': 0.5,
        'total_steps': 10000000,
        'modeltype': 'ANI1xnr',
        'friction': 0.1,
        'logfile': 'auto', # '-' for stdout
        'loginterval': 100,
        'trajfile': 'auto',
        'trajinterval': 10,
        'logger': 'default',
        # optimizations setting
        'optimize_geometry': False,
        'opt_fmax': 0.1,  # Maximum force criterion in eV/Angstrom
        'max_opt_steps': 1000,  # Maximum optimization steps
        'optimizer': None
    }
    def __init__(self, concentration:float, system_filepath:str, **kwargs):
        """
        Initializes the OverlayMolecules instance with optional file names and parameters.

        Parameters
        ----------
        concentration : float
            carbon concentration in g/cc
        system_filepath : str
            filepath of initial system in xyz format
        **kwargs : keyword arguments
            Optional parameters
        """
        self.concentration = concentration
        self.system_filepath = system_filepath 
        self.simulated_steps = 0
        self.continued = False
        # default parameters
        self.parameters = Parameters({**NanoReactor.default_parameters, **kwargs})

        # default name if trajfile/logfile is not provided
        self.id = f'conc_{self.concentration}_friction_{self.parameters.friction}_temperature_{self.parameters.temperature_K}'
        if self.parameters.trajfile == 'auto':
            self.parameters.trajfile = self.id + '.traj'
        if self.parameters.logfile == 'auto':
            self.parameters.logfile = self.id + '.log'

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load ani calculator
        self.calc = animodel(modelname=self.parameters.modeltype)

        # read system xyz file
        self.system = read(self.system_filepath)
        
        # set box length
        self.box_length_nm = get_box_length_nm(
            concentration=self.concentration, 
            num_molecules=len(self.system) // 4
            )
        # set periodic boundary conditions
        self.cell = [self.box_length_nm * 10] * 3 # in A
        self.pbc = [True] * 3
        self.system.cell = self.cell
        self.system.pbc = self.pbc

        # set system calculatior 
        self.system.calc = self.calc

        # total simulation time
        self.simulation_time_ns = self.parameters.time_step_fs * self.parameters.total_steps / 1e6 # in ns


    def get_box_length_nm(self, concentration:float, num_molecules:int):
        """return box length based on concentration and number of atoms
        """
        Na = 6.022140857 # avogadro number pre-exponent
        M_ac = 26.0 # acetylene molar weight
        return ((num_molecules * M_ac) / (concentration * Na * 100))**(1/3) # in nm

    def animodel(self, modelname:str):
        """self torchani model calculator
        """
        if modelname == 'ANI1xnr':
            return torchani.models.ANI1xnr().to(self.device).ase()
        elif modelname == 'ANI1ccx':
            return torchani.models.ANI1ccx().to(self.device).ase()
        elif modelname == 'ANI2x':
            return torchani.models.ANI2x().to(self.device).ase()
        elif modelname == 'ANI1x':
            return torchani.models.ANI1x().to(self.device).ase()
        else:
            raise ValueError(f"Model {modelname} Not Found")

    def printenergy(self) -> None:
        """Function to print the potential, kinetic and total energy"""
        atoms = self.system
        # update simulated steps
        if not self.simulated_steps:
            self.simulated_steps = 1
        self.simulated_steps += self.parameters.loginterval

        running_time = time.time() - self.start_time
        if self.continued:
            time_per_steps = running_time / (self.simulated_steps - self.previous_steps)
            remaining_time = (self.parameters.total_steps - (self.simulated_steps - self.previous_steps)) * time_per_steps
        else:
            time_per_steps = running_time / self.simulated_steps
            remaining_time = (self.parameters.total_steps - self.simulated_steps) * time_per_steps

        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        temperature = ekin / (1.5 * units.kB)

        print(f'Energy per atom: Epot = {epot:.3f}eV  Ekin = {ekin:.3f}eV '
              f'(T={temperature:3.0f}K)  Etot = {epot+ekin:.3f}eV '
              '| Avg Time per step = ',f'\033[1;32m{time_per_steps:.3f}s  \033[0m  '
              'Remaining Time = ',f'\033[1;31m{remaining_time/3600:.2f}h\033[0m')
        # print(f'Energy per atom: Epot = {epot:.3f}eV  Ekin = {ekin:.3f}eV '
        #       f'(T={temperature:3.0f}K)  Etot = {epot+ekin:.3f}eV')
        # print(f'Avg Time per step = {time_per_steps:.3f}s  Remaining Time = {remaining_time/360:.2f}h\n')

    def simulation_info(self):
        return f"""
    ======================================================
    System Info.
    ------------------------------------------------------
    N_molecules : {len(self.system) // 4}
    N_atoms     : {len(self.system)}
    Conc.       : {self.concentration}  g/cc
    Box Len.    : {self.box_length_nm:.3f}  nm
    PBC         : {self.pbc}
    ------------------------------------------------------
    
    ======================================================
    Simulation Info.
    ------------------------------------------------------
    MLP model   : {self.parameters.modeltype}
    Temperature : {self.parameters.temperature_K}  K
    Time Step   : {self.parameters.time_step_fs}  fs
    Total Steps : {self.parameters.total_steps} times
    Sim. Time   : {self.simulation_time_ns:.3f} ns
    friction    : {self.parameters.friction} 1/fs
    Device      : {self.device}
    ------------------------------------------------------
    
    ======================================================
    Results
    ------------------------------------------------------
    log interval: {self.parameters.loginterval} steps
    trj interval: {self.parameters.trajinterval} steps
    logfile     : {self.parameters.logfile}
    trajfile    : {self.parameters.trajfile}
    ------------------------------------------------------"""


    def optimize_geometry(self):
        """optimize before MD simulation
        """
        if self.parameters.optimizer:
            self.optimizer = self.parameters.optimizer
        else:
            from ase.optimize import FIRE
            warnings.warn("Optimizer is not given.\n -> Use default Optimizer : FIRE", category=UserWarning)
            self.optimizer = FIRE

        opt = self.optimizer(self.system) 
        
        try:
            assert opt.run(fmax=self.parameters.opt_fmax, steps=self.parameters.max_opt_steps), 'Optimization failed'
            return True
        except Exception as e:
            print(e)
            return False




    def run_simulation(self):
        """run simulation
        """
        self.start_time = time.time()
        with open(f"simulation_info_{self.id}.txt", 'w') as file:
            file.write(self.simulation_info())
        
        # Geometry optimization
        if self.parameters.optimize_geometry:
            if not self.optimize_geometry(): " Stop Simulation "
            else:
                clear_output()
                print("Optimization normally terminated.\n")

        MaxwellBoltzmannDistribution(
        atoms=self.system,
        temperature_K=self.parameters.temperature_K
        )

        self.parameters.time_step_fs *= units.fs


        dyn = Langevin(
            atoms=self.system,
            timestep=self.parameters.time_step_fs,
            friction=self.parameters.friction,
            temperature_K=self.parameters.temperature_K,
            logfile=self.parameters.logfile,
            loginterval=self.parameters.loginterval
            )

        traj = Trajectory(self.parameters.trajfile, 'w', self.system)
        dyn.attach(traj.write, interval=self.parameters.trajinterval)

        if self.parameters.logger == 'default':
            dyn.attach(self.printenergy, interval=self.parameters.loginterval)
        elif self.parameters.logger:
            dyn.attach(self.parameters.logger, interval=self.parameters.loginterval)

        dyn.run(steps=self.parameters.total_steps)


    def continue_simulation(self, traj_filepath: str):
        """
        Continue simulation from a trajectory file

        Parameters
        ----------
        traj_filepath : str
            Path to the trajectory file to continue from and append to
        """
        self.start_time = time.time()
        self.continued = True
        # Read the trajectory to count frames
        traj = Trajectory(traj_filepath)
        completed_steps = (len(traj) - 1) * self.parameters.trajinterval
        self.previous_steps = completed_steps
        self.parameters.total_steps -= completed_steps
        self.simulated_steps = completed_steps

        print(f"Completed steps: {self.simulated_steps}")
        print(f"Remaining steps: {self.parameters.total_steps}")

        # Read the last frame from the trajectory
        self.system = traj[-1]

        # Set up the system again
        self.system.cell = self.cell
        self.system.pbc = self.pbc
        self.system.calc = self.calc

        # Ensure time_step is in correct units
        self.parameters.time_step_fs *= units.fs

        # Set up dynamics
        dyn = Langevin(
            atoms=self.system,
            timestep=self.parameters.time_step_fs,
            friction=self.parameters.friction,
            temperature_K=self.parameters.temperature_K,
            logfile=self.parameters.logfile,  # 원래 log 파일 사용
            loginterval=self.parameters.loginterval
        )

        # 기존 파일에 append 모드로 추가
        traj = Trajectory(traj_filepath, 'a', self.system)
        dyn.attach(traj.write, interval=self.parameters.trajinterval)

        if self.parameters.logger == 'default':
            dyn.attach(self.printenergy, interval=self.parameters.loginterval)
        elif self.parameters.logger:
            dyn.attach(self.parameters.logger, interval=self.parameters.loginterval)

        # Run the remaining steps
        dyn.run(steps=self.parameters.total_steps)
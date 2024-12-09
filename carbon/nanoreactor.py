import numpy as np
import torch
import torchani
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


def randomized_xyz_builder(box_size, min_distance, num_points, max_trials=10000):
    """grid based random initial system builder
    """
    # count number of grids
    grid_count = (box_size // min_distance) ** 3
    if num_points > grid_count:
        raise ValueError("num_points can't larger than number of grids")

    points = []
    grid_size = min_distance
    grid = {}
    trials = 0

    while len(points) < num_points:
        # generate random point
        point = (
            random.uniform(0, box_size),
            random.uniform(0, box_size),
            random.uniform(0, box_size)
        )

        # calculate grid coordinate of random point
        grid_coord = (
            int(point[0] // grid_size),
            int(point[1] // grid_size),
            int(point[2] // grid_size)
        )
        # add point if no points in that grid
        if grid_coord not in grid:
            points.append(point)
            grid[grid_coord] = point
            trials = 0
        else:
            trials += 1

        # return error when max trials exceed
        if trials > max_trials: raise RuntimeError("Max trials exceeded")

    return points
    
Na = 6.022140857 # avogadro number pre-exponent
Mc = 12.0 # 12C carbon atomic mass
get_box_length_nm = lambda concentration, num_atoms: ((num_atoms * Mc) / (concentration * Na * 100))**(1/3) # in nm


def save_xyz(points, symbols='C', filename='random.xyz'):
    """
    Save points to xyz file, when it is homogeneous system.

    Parameters
    ----------
    - points : xyz coordinate in nm unit
    - symbols : atom symbols
    - filename : output file name
    """
    xs, ys, zs = zip(*points)
    with open(filename, 'w') as file:
        num_atoms = len(xs)
        file.write(f"{num_atoms}\n")
        file.write("\n")
        for x, y, z in zip(xs, ys, zs):
            file.write(f"{symbols} {x * 10:10.6f} {y * 10:10.6f} {z * 10:10.6f}\n")


def visualize_xyz(atom, box_size_nm=False, group=None, save_name=None):
    positions = atom.get_positions()
    if group:
        positions = positions[group]
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

    if box_size_nm:

        # Box info
        n = box_size_nm*10
        x_coords = [0, n, n, 0, 0, n, n, 0]
        y_coords = [0, 0, n, n, 0, 0, n, n]
        z_coords = [0, 0, 0, 0, n, n, n, n]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 아래 네 모서리
            (4, 5), (5, 6), (6, 7), (7, 4),  # 위 네 모서리
            (0, 4), (1, 5), (2, 6), (3, 7)   # 위아래 연결 모서리
        ]

        x_lines = []
        y_lines = []
        z_lines = []

        for edge in edges:
            for vertex in edge:
                x_lines.append(x_coords[vertex])
                y_lines.append(y_coords[vertex])
                z_lines.append(z_coords[vertex])
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

    plots = data=[
        # atoms
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(size=1, color='gray')
        ),
        # scaling bar
        go.Scatter3d(
            x=[0, 10],
            y=[0, 0],
            z=[0, 0],
            mode='lines+text',
            line=dict(color='red', width=10),
            text=['1 nm'],
            textposition='top center',
            textfont=dict(size=12, color='red'),
            showlegend=False
        )]
    if box_size_nm:
        plots.append(
            go.Scatter3d(
                    x=x_lines,
                    y=y_lines,
                    z=z_lines,
                    mode='lines',
                    line=dict(color='grey', width=3),
                    showlegend=False
                    )
        )
    # plot
    fig = go.Figure(
        plots
    )

    # update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showline=False, title='', visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, title='', visible=False),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, title='', visible=False),
            bgcolor='black'
        ),
        title=f'Num. atoms {len(atom)}',
        paper_bgcolor='black'
    )

    fig.update_scenes(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        zaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        zaxis_zeroline=False,
    )

    fig.show()
    if save_name:
        fig.write_html(f'{save_name}.html')


def initial_system_builder(**params):
    """
    """
    # params
    conc = params.get('concentration')
    assert conc, "concentration is not given"
    min_distance_Ang = params.get('min_distance_Ang', 1.7)
    num_atoms = params.get('num_atoms', 5000)
    max_trials = params.get('max_trials', 10000)
    save = params.get('save', False)
    display_system = params.get('display_system', False)
    filename = params.get('filename', 'auto')

    # reset filename
    if filename == 'auto':
        filename = f'box_{conc}[gcc-1]_points_{num_atoms}[atoms].xyz'

    #box size
    box_size_nm = get_box_length_nm(concentration=conc, num_atoms=num_atoms) # in nm

    points = randomized_xyz_builder(
        box_size=box_size_nm, 
        min_distance=min_distance_Ang/10, # convert Angs to nm
        num_points=num_atoms, 
        max_trials=max_trials
        ) # in nm scale
    
    if save:
        save_xyz(
            points=points, 
            filename=filename,
            symbols='C'
            )
    if display_system:
        system = read(filename)
        visualize_xyz(system, box_size_nm=box_size_nm)


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
        'optimize_geometry': True,
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
            num_atoms=len(self.system)
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


    def get_box_length_nm(self, concentration:float, num_atoms:int):
        """return box length based on concentration and number of atoms
        """
        Na = 6.022140857 # avogadro number pre-exponent
        Mc = 12.0 # 12C carbon atomic mass
        return ((num_atoms * Mc) / (concentration * Na * 100))**(1/3) # in nm

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
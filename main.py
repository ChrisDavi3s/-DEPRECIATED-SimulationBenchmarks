import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, List
from enum import Enum
import os
import re
from ase.io import read
from nequip.ase import NequIPCalculator
from tqdm import tqdm

class Method:
    def __init__(self, name):
        self.name = name

    def add_data(self, trajectory):
        raise NotImplementedError("Subclasses must implement the add_data method.")

class VASPMethod(Method):
    def __init__(self, name, directory, base_name):
        super().__init__(name)
        self.directory = directory
        self.base_name = base_name

    def add_data(self, trajectory):
        pattern = re.compile(f"{self.base_name}_(\\d+)\\.xml")
        vasp_files = [f for f in os.listdir(self.directory) if pattern.match(f)]

        def get_frame_number(filename):
            match = pattern.match(filename)
            return int(match.group(1)) if match else -1

        sorted_vasp_files = sorted(vasp_files, key=get_frame_number)

        for filename in tqdm(sorted_vasp_files, desc=f"Adding {self.name} data"):
            vasp_atom = read(os.path.join(self.directory, filename))
            frame_number = get_frame_number(filename)
            trajectory.frames[frame_number].info[f'{self.name}_total_energy'] = vasp_atom.get_potential_energy()
            trajectory.frames[frame_number].arrays[f'{self.name}_forces'] = vasp_atom.get_forces()
            trajectory.frames[frame_number].info[f'{self.name}_stress'] = vasp_atom.get_stress()

class NequIPMethod(Method):
    def __init__(self, name, model_path):
        super().__init__(name)
        self.model_path = model_path

    def add_data(self, trajectory):
        calc = NequIPCalculator.from_deployed_model(self.model_path)
        for atom in tqdm(trajectory.frames, desc=f"Adding {self.name} data"):
            atom.calc = calc
            atom.info[f'{self.name}_total_energy'] = atom.get_potential_energy()
            atom.arrays[f'{self.name}_forces'] = atom.get_forces()
            atom.info[f'{self.name}_stress'] = atom.get_stress()

class Frames:
    '''
    Class to store 
    '''
    def __init__(self, xyz_file):
        self.xyz_file = xyz_file
        self.frames = read(self.xyz_file, index=':')

    def add_method_data(self, method):
        method.add_data(self)

class ComparisonParams:
    def __init__(self, calc_energy_metrics=True, calc_forces_metrics=True, calc_stress_metrics=True,
                 plot_energy=True, plot_forces=True, plot_stress=True):
        self.calc_energy_metrics = calc_energy_metrics
        self.calc_forces_metrics = calc_forces_metrics
        self.calc_stress_metrics = calc_stress_metrics
        self.plot_energy = plot_energy
        self.plot_forces = plot_forces
        self.plot_stress = plot_stress

class DataType(Enum):
    ENERGY_PER_STRUCTURE = ('energy_per_structure', 'eV/structure')
    ENERGY_PER_ATOM = ('energy_per_atom', 'eV/atom')
    FORCES = ('forces', 'eV/Å')
    TOTAL_FORCES = ('total_forces', 'eV/Å')
    STRESS = ('stress', 'eV/Å³')

    def __init__(self, key, unit):
        self.key = key
        self.unit = unit

class MetricsResult:
    def __init__(self, mae: float, rmse: float, correlation: float):
        self.mae = mae
        self.rmse = rmse
        self.correlation = correlation

class Comparer:
    @staticmethod
    def compare(trajectory: Frames, method1: Method, method2: Method, params: ComparisonParams) -> Tuple[Dict[str, MetricsResult], Dict[str, Dict[str, MetricsResult]], MetricsResult]:
        """
        Compare two methods across a trajectory.

        Parameters:
        trajectory (Trajectory): The trajectory to compare.
        method1 (Method): The first method to compare.
        method2 (Method): The second method to compare.
        params (ComparisonParams): Parameters for the comparison.

        Returns:
        Tuple[Dict[str, MetricsResult], Dict[str, Dict[str, MetricsResult]], MetricsResult]:
            - Energy metrics (per structure and per atom)
            - Forces metrics (per atom type)
            - Stress metrics
        """
        energy_metrics = {}
        forces_metrics = {}
        stress_metrics = None

        if params.calc_energy_metrics or params.plot_energy:
            method1_energies, method2_energies, num_atoms = Comparer._extract_energies(trajectory, method1, method2)
            energy_metrics['per_structure'] = Comparer._calculate_metrics(method1_energies, method2_energies)
            energy_metrics['per_atom'] = Comparer._calculate_metrics(method1_energies / num_atoms, method2_energies / num_atoms)

        if params.calc_forces_metrics or params.plot_forces:
            method1_forces, method2_forces = Comparer._extract_forces(trajectory, method1, method2)
            forces_metrics = {atom_type: Comparer._calculate_metrics(np.array(method1_forces[atom_type]), np.array(method2_forces[atom_type]))
                              for atom_type in method1_forces.keys()}

        if params.calc_stress_metrics or params.plot_stress:
            method1_stresses, method2_stresses = Comparer._extract_stresses(trajectory, method1, method2)
            stress_metrics = Comparer._calculate_metrics(method1_stresses, method2_stresses)

        return energy_metrics, forces_metrics, stress_metrics

    @staticmethod
    def plot(trajectory: Frames, method1: Method, method2: Method, params: ComparisonParams, 
             energy_metrics: Dict[str, MetricsResult], forces_metrics: Dict[str, Dict[str, MetricsResult]], 
             stress_metrics: MetricsResult, display_correlation: bool = True) -> None:
        """
        Plot comparison results.

        Parameters:
        trajectory (Trajectory): The trajectory used for comparison.
        method1 (Method): The first method used in the comparison.
        method2 (Method): The second method used in the comparison.
        params (ComparisonParams): Parameters used for the comparison.
        energy_metrics (Dict[str, MetricsResult]): Energy comparison metrics.
        forces_metrics (Dict[str, Dict[str, MetricsResult]]): Forces comparison metrics.
        stress_metrics (MetricsResult): Stress comparison metrics.
        display_correlation (bool): Whether to display correlation in plots.

        Returns:
        None
        """
        if params.plot_energy and energy_metrics:
            method1_energies, method2_energies, num_atoms = Comparer._extract_energies(trajectory, method1, method2)
            
            # Per-structure energy plots
            Comparer._plot_data(method1_energies, method2_energies, method1, method2, DataType.ENERGY_PER_STRUCTURE, energy_metrics['per_structure'], display_correlation)
            Comparer._plot_mae_boxplot(np.abs(method1_energies - method2_energies), method1, method2, DataType.ENERGY_PER_STRUCTURE)
            
            # Per-atom energy plots
            method1_energies_per_atom = method1_energies / num_atoms
            method2_energies_per_atom = method2_energies / num_atoms
            Comparer._plot_mae_boxplot(np.abs(method1_energies_per_atom - method2_energies_per_atom), method1, method2, DataType.ENERGY_PER_ATOM)

        if params.plot_forces and forces_metrics:
            method1_forces, method2_forces = Comparer._extract_forces(trajectory, method1, method2)
            mae_per_atom_type = {atom_type: np.abs(np.array(method1_forces[atom_type]) - np.array(method2_forces[atom_type])) 
                                 for atom_type in method1_forces.keys()}
            Comparer._plot_mae_per_atom_boxplot(mae_per_atom_type, method1, method2, DataType.FORCES)

        if params.plot_stress and stress_metrics:
            method1_stresses, method2_stresses = Comparer._extract_stresses(trajectory, method1, method2)
            Comparer._plot_data(method1_stresses, method2_stresses, method1, method2, DataType.STRESS, stress_metrics, display_correlation)
            Comparer._plot_mae_boxplot(np.abs(method1_stresses - method2_stresses), method1, method2, DataType.STRESS)

    @staticmethod
    def _extract_energies(trajectory: Frames, method1: Method, method2: Method) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract energies from trajectory for two methods.

        Parameters:
        trajectory (Trajectory): The trajectory to extract energies from.
        method1 (Method): The first method.
        method2 (Method): The second method.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Energies for method1, energies for method2, and number of atoms per frame.
        """
        method1_energies = np.array([atom.info[f'{method1.name}_total_energy'] for atom in trajectory.frames])
        method2_energies = np.array([atom.info[f'{method2.name}_total_energy'] for atom in trajectory.frames])
        num_atoms = np.array([len(frame) for frame in trajectory.frames])
        return method1_energies, method2_energies, num_atoms
     
    @staticmethod
    def print_metrics(energy_metrics: Dict[str, MetricsResult], forces_metrics: Dict[str, Dict[str, MetricsResult]], 
                      stress_metrics: MetricsResult) -> None:
        """
        Print comparison metrics.

        Parameters:
        energy_metrics (Dict[str, MetricsResult]): Energy comparison metrics.
        forces_metrics (Dict[str, Dict[str, MetricsResult]]): Forces comparison metrics.
        stress_metrics (MetricsResult): Stress comparison metrics.

        Returns:
        None
        """
        print("\n" + "="*40)
        print("Metrics Summary")
        print("="*40)

        if energy_metrics:
            print("\nEnergy Metrics:")
            print("-" * 20)
            print("Per Structure:")
            Comparer._print_metric_result(energy_metrics['per_structure'], DataType.ENERGY_PER_STRUCTURE)
            print("\nPer Atom:")
            Comparer._print_metric_result(energy_metrics['per_atom'], DataType.ENERGY_PER_ATOM)

        if forces_metrics:
            print("\nForces Metrics:")
            print("-" * 20)
            for atom_type, metrics in forces_metrics.items():
                print(f"\nAtom Type: {atom_type}")
                Comparer._print_metric_result(metrics, DataType.FORCES)

            # Calculate and print total forces metrics
            total_mae = np.mean([metrics.mae for metrics in forces_metrics.values()])
            total_rmse = np.sqrt(np.mean([metrics.rmse**2 for metrics in forces_metrics.values()]))
            total_correlation = np.mean([metrics.correlation for metrics in forces_metrics.values()])
            
            print("\nTotal Forces Metrics:")
            print("-" * 20)
            print(f"MAE: {total_mae:.6f} {DataType.FORCES.unit}")
            print(f"RMSE: {total_rmse:.6f} {DataType.FORCES.unit}")
            print(f"Average Correlation: {total_correlation:.6f}")


        if stress_metrics:
            print("\nStress Metrics:")
            print("-" * 20)
            Comparer._print_metric_result(stress_metrics, DataType.STRESS)

    @staticmethod
    def _calculate_metrics(reference: np.ndarray, predicted: np.ndarray) -> MetricsResult:
        """
        Calculate comparison metrics.

        Parameters:
        reference (np.ndarray): Reference data.
        predicted (np.ndarray): Predicted data.

        Returns:
        MetricsResult: Calculated metrics (MAE, RMSE, correlation).
        """
        mae = np.mean(np.abs(reference - predicted))
        rmse = np.sqrt(np.mean((reference - predicted)**2))
        correlation = np.corrcoef(reference, predicted)[0, 1]
        return MetricsResult(mae, rmse, correlation)

    @staticmethod
    def _print_metric_result(metrics: MetricsResult, data_type: DataType) -> None:
        """
        Print a single metric result.

        Parameters:
        metrics (MetricsResult): The metrics to print.
        data_type (DataType): The type of data for the metrics.

        Returns:
        None
        """
        print(f"MAE: {metrics.mae:.6f} {data_type.unit}")
        print(f"RMSE: {metrics.rmse:.6f} {data_type.unit}")
        print(f"Correlation: {metrics.correlation:.6f}")

    @staticmethod
    def _extract_energies(trajectory: Frames, method1: Method, method2: Method) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract energies from trajectory for two methods.

        Parameters:
        trajectory (Trajectory): The trajectory to extract energies from.
        method1 (Method): The first method.
        method2 (Method): The second method.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Energies for method1, energies for method2, and number of atoms per frame.
        """
        method1_energies = np.array([atom.info[f'{method1.name}_total_energy'] for atom in trajectory.frames])
        method2_energies = np.array([atom.info[f'{method2.name}_total_energy'] for atom in trajectory.frames])
        num_atoms = np.array([len(frame) for frame in trajectory.frames])
        return method1_energies, method2_energies, num_atoms

    @staticmethod
    def _extract_forces(trajectory: Frames, method1: Method, method2: Method) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
        """
        Extract forces from trajectory for two methods.

        Parameters:
        trajectory (Trajectory): The trajectory to extract forces from.
        method1 (Method): The first method.
        method2 (Method): The second method.

        Returns:
        Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]: Forces for method1 and method2, organized by atom type.
        """
        method1_forces = {}
        method2_forces = {}
        for atom in trajectory.frames:
            forces1 = atom.arrays[f'{method1.name}_forces']
            forces2 = atom.arrays[f'{method2.name}_forces']
            for i, symbol in enumerate(atom.get_chemical_symbols()):
                if symbol not in method1_forces:
                    method1_forces[symbol] = []
                    method2_forces[symbol] = []
                method1_forces[symbol].append(forces1[i])
                method2_forces[symbol].append(forces2[i])
        return method1_forces, method2_forces

    @staticmethod
    def _extract_stresses(trajectory: Frames, method1: Method, method2: Method) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract stresses from trajectory for two methods.

        Parameters:
        trajectory (Trajectory): The trajectory to extract stresses from.
        method1 (Method): The first method.
        method2 (Method): The second method.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Stresses for method1 and method2.
        """
        method1_stresses = np.array([atom.info[f'{method1.name}_stress'] for atom in trajectory.frames])
        method2_stresses = np.array([atom.info[f'{method2.name}_stress'] for atom in trajectory.frames])
        return method1_stresses.flatten(), method2_stresses.flatten()

    @staticmethod
    def _calculate_metrics(reference: np.ndarray, predicted: np.ndarray) -> MetricsResult:
        '''
        Calculate comparison metrics.

        Parameters:
        reference (np.ndarray): Reference data.
        predicted (np.ndarray): Predicted data.

        Returns:
        MetricsResult: Calculated metrics (MAE, RMSE, correlation).
        '''
        mae = np.mean(np.abs(reference - predicted))
        rmse = np.sqrt(np.mean((reference - predicted)**2))
        correlation = np.corrcoef(reference, predicted)[0, 1]
        return MetricsResult(mae, rmse, correlation)

    @staticmethod
    def _plot_data(method1_data: np.ndarray, method2_data: np.ndarray, method1: Method, method2: Method, 
                   data_type: DataType, metrics: MetricsResult, display_correlation: bool) -> None:
        """
        Plot comparison data.

        Parameters:
        method1_data (np.ndarray): Data from the first method.
        method2_data (np.ndarray): Data from the second method.
        method1 (Method): The first method.
        method2 (Method): The second method.
        data_type (DataType): The type of data being plotted.
        metrics (MetricsResult): Metrics for the comparison.
        display_correlation (bool): Whether to display correlation in the plot.

        Returns:
        None
        """
        plt.figure(figsize=(6, 5))
        plt.scatter(method1_data, method2_data, alpha=0.6, color='black', s=20, edgecolors='none')
        
        # Determine axis limits
        all_data = np.concatenate([method1_data, method2_data])
        min_val, max_val = np.min(all_data), np.max(all_data)
        range_val = max_val - min_val
        buffer = range_val * 0.05  # 5% buffer
        plot_min, plot_max = min_val - buffer, max_val + buffer

        # Add identity line
        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=1)

        # Set labels and title
        plt.xlabel(f'{method1.name} {data_type.key.replace("_", " ").title()} ({data_type.unit})', fontsize=10)
        plt.ylabel(f'{method2.name} {data_type.key.replace("_", " ").title()} ({data_type.unit})', fontsize=10)
        plt.title(f'{data_type.key.replace("_", " ").title()} Comparison', fontsize=12)

        if display_correlation:
            stats_text = (f"R: {metrics.correlation:.4f}\n"
                          f"MAE: {metrics.mae:.4f} {data_type.unit}\n"
                          f"RMSE: {metrics.rmse:.4f} {data_type.unit}")
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8))

        plt.tight_layout()
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plt.xlim(plot_min, plot_max)
        plt.ylim(plot_min, plot_max)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.show()

    @staticmethod
    def _plot_mae_boxplot(mae_data: np.ndarray, method1: Method, method2: Method, data_type: DataType) -> None:
        """
        Plot MAE boxplot.

        Parameters:
        mae_data (np.ndarray): MAE data to plot.
        method1 (Method): The first method.
        method2 (Method): The second method.
        data_type (DataType): The type of data being plotted.

        Returns:
        None
        """

        plt.figure(figsize=(4, 5))
        box = plt.boxplot([mae_data.flatten()], patch_artist=True, widths=0.6)
        
        # Set colors to grayscale
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box[element], color='black')
        for patch in box['boxes']:
            patch.set_facecolor('white')
        
        plt.xticks([1], [f"{method2.name}"], fontsize=10)
        
        if data_type == DataType.ENERGY_PER_STRUCTURE:
            ylabel = 'Energy MAE (eV/structure)'
        elif data_type == DataType.ENERGY_PER_ATOM:
            ylabel = 'Energy MAE (eV/atom)'
        elif data_type == DataType.FORCES:
            ylabel = 'Forces MAE (eV/Å)'
        elif data_type == DataType.STRESS:
            ylabel = 'Stress MAE (eV/Å³)'
        else:
            ylabel = f'MAE ({data_type.unit})'
        
        plt.ylabel(ylabel, fontsize=10)
        title = f"{data_type.key.replace('_', ' ').title()} MAE"
        plt.title(title, fontsize=12)
        
        plt.tight_layout()
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.show()

    @staticmethod
    def _plot_mae_per_atom_boxplot(mae_data: Dict[str, np.ndarray], method1: Method, method2: Method, data_type: DataType) -> None:
        """
        Plot MAE per atom type boxplot.

        Parameters:
        mae_data (Dict[str, np.ndarray]): MAE data for each atom type.
        method1 (Method): The first method.
        method2 (Method): The second method.
        data_type (DataType): The type of data being plotted.

        Returns:
        None
        """
        plt.figure(figsize=(6, 5))
        atom_types = list(mae_data.keys())
        filtered_data = [mae_data[atom_type].flatten() for atom_type in atom_types]
        box = plt.boxplot(filtered_data, patch_artist=True)
        
        # Set colors to grayscale
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box[element], color='black')
        for patch in box['boxes']:
            patch.set_facecolor('white')
        
        plt.xticks(range(1, len(atom_types) + 1), atom_types, fontsize=10, rotation=45, ha='right')
        plt.xlabel('Atom Type', fontsize=10)
        plt.ylabel(f'MAE ({data_type.unit})', fontsize=10)
        title = f"{data_type.key.replace('_', ' ').title()} MAE per Atom Type"
        plt.title(title, fontsize=12)
        plt.tight_layout()
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.show()        

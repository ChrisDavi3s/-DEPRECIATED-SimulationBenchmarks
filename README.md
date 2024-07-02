# SimulationBenchmars


Usage:

```python
xyz_file = 'random_32_Li6PS5Cl0.5Br0.5_221_800K_PBE.extxyz'
frames = Frames(xyz_file)

vasp_method = VASPMethod('DFT (PBE)', '32_frames_ClBr_vaspout', 'vasprun_frame')
nequip_method = NequIPMethod('Allegro', 'deployed.pth')

frames.add_method_data(vasp_method)
frames.add_method_data(nequip_method)

params = ComparisonParams(calc_energy_metrics=True,
                          calc_forces_metrics=True,
                          calc_stress_metrics=True,
                          plot_energy=True,
                          plot_forces=True,
                          plot_stress=True)

energy_metrics, forces_metrics, stress_metrics = Comparer.compare(frames, vasp_method, nequip_method, params)
Comparer.print_metrics(energy_metrics, forces_metrics, stress_metrics)
Comparer.plot(frames, vasp_method, nequip_method, params, energy_metrics, forces_metrics, stress_metrics)
```

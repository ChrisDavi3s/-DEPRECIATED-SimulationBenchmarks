# SimulationBenchmarks


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

Energies:

![image](https://github.com/ChrisDavi3s/SimulationBenchmarks/assets/9642076/37a093cb-70b9-4fdc-9697-c567d455b952)

Forces:

![image](https://github.com/ChrisDavi3s/SimulationBenchmarks/assets/9642076/f0daa71b-5038-4b39-be8c-39aac25e1450)

Stresses:

![image](https://github.com/ChrisDavi3s/SimulationBenchmarks/assets/9642076/2086c236-6dac-4281-aae4-d2b18514b257)

All come with appropriate box plots.

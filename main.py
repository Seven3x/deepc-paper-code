import numpy as np
from Controllers.deepc import DeePC
from Controllers.lqr import LQR
from quadcopter import Quadcopter
from Simulator.simulation import Simulation, SimulationPlotter
from trajectory_generator import TrajectoryGenerator
from hdf5_reader import HDF5Reader
from visualization import QuadcopterAnimator

h = 0.1 # Sampling time
reference_duration = 13

system = Quadcopter(h=h)
reader = HDF5Reader("DeePC_Quadcopter/Results")

running = True

while running:
    
    a = input("Animate result, simulate or exit (a/s/e): ")

    if a == 'a':
        data, chosen_file = reader.run()
        animator = QuadcopterAnimator(system)
        animator.animate(data, chosen_file+'.mp4')

    elif a == 's':
        # sort = 'figure8', 'constant', 'step', 'box'
        trajectory = TrajectoryGenerator(sort="step",
                                        system=system,
                                        duration=reference_duration,
                                        has_initial_ref=True)
        initial_controller = LQR(system, noise=0.1, seed=42)
        controller = DeePC(system,
                        trajectory,
                        initial_controller,
                        is_regularized=True)
        t_final = controller.T*h + reference_duration
        sim = Simulation(system, controller, t_final=t_final)
        plotter = SimulationPlotter(system)

        result = sim.simulate()
        plotter.plot(result, trajectory)

        a = input("Would you like to save the result? (y/n): ")
        if a == 'y':
            info = input("Enter additional information to save with the result: ")
            reader.save_to_hdf5(result, f"deepc_quadcopter_{trajectory.sort}_{info}")
    
    elif a == 'e':
        running = False 
    
    else:
        print("Invalid input. Please enter 'a' to animate or 's' to simulate.")
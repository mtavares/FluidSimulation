"""
Fluid Simulation based on:
https://thecodingtrain.com/CodingChallenges/132-fluid-simulation.html
https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

MIT License
"""
import FluidSquare

fluid_square = FluidSquare.FluidSquare(diffusion=0, viscosity=0.0000001, dt=1)
fluid_square.run()
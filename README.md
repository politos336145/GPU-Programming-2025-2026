# Angry Santa GPU Snowball Simulation

We are approaching the Christmas season and Santa Claus is getting tired and angry because his elves team is behind schedule. In a moment of pure frustration, he decides to destroy the entire village with a gigantic rolling snowball.

So… let’s simulate this physical phenomenon.

![Physics 1](./Physics%201.png)
![Physics 2](./Physics%202.png)


In the project, we model the motion of a rigid body (for simplicity, approximated as a perfect sphere) with variable mass sliding down an inclined plane under Newtonian dynamics. The interesting part is the varying mass: the snowball grows over time, and this behavior can be efficiently simulated through GPU parallelism.

## System Overview

We start from a 2D grid (top view) that represents the terrain.

- At every discrete time step, the CPU computes sphere physics parameters (mass, radius, speed…) and each cell of the grid checks whether it is currently covered by the snowball (via GPU).
- If it is, that cell launches a second layer of computation: a finer N×M sub-grid that represents the set of snow particles in that area.
- Using a stochastic process, each particle decides whether it sticks to the snowball or not.

---

## GPU-Related Concepts Covered

This project touches several important GPU-related concepts:

- **Multi-level thread parallelism via CUDA Streams**, allowing independent parts of the simulation to run asynchronously.
- **Atomic reduction operations**, used to safely accumulate mass contributions from thousands of concurrent threads.
- **CPU vs GPU comparison**:  
  A CPU-only approach results in a naive *O(n)* algorithm, while a GPU implementation uses thousands of threads in parallel, dramatically improving performance.
- **Memory Optimization Techniques**:
  - Shared Memory Usage  
  - Coalesced Memory Access  
  - Occupancy Optimization  
  - Reductions  

- **Multi-GPU Scalability**:  
  The terrain can be divided into sub-areas (A, B, C, D, …), each handled by a different GPU.  
  This virtually removes scalability limits and enables very large-scale simulations.

---

## Broader Idea

The broader idea behind this project is to explore how a graphics engine (similar to those used in modern video games) might implement a specialized physics function by fully leveraging GPU computing power.
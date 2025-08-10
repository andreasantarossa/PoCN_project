# Physics of Complex Networks Project

## Introduction
This is the final project for the *Physics of Complex Networks* course, part of the **Physics of Data** Master’s degree program at the University of Padua (UNIPD).  
The project consists of two tasks, covering both theoretical and practical aspects of complex networks:

- **Dynamics on networks** — Simulation of epidemic spreading on static networks using SI, SIS, SIR, and SEIR models, in both homogeneous and heterogeneous mean-field approximations.  
- **Public transport in large cities worldwide** — The dataset records the development of public transportation networks over time (different modes, e.g., tram, train, subway, etc., and potentially multiple lines per mode, such as distinct subway lines).  
  The task is to generate, for each city in the database, two files:  
  - **Nodes**: `nodeID, nodeLabel, latitude, longitude, mode, year`  
  - **Edges**: `nodeID_from, nodeID_to, mode, line, year`

## Repository structure
The repository is organized into three main parts:

- **code** — Contains all the code for both tasks, including simulations, network generation, data analysis, and visualization tools.  
- **data** — Stores the generated networks and simulation outputs for each task.  
- **latex** — Contains all text and image files required to reproduce the final report.

Additionally, the main folder includes a detailed report describing the methodology, results, and conclusions for each task.

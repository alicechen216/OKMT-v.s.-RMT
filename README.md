# Occupation Kernel vs. RMT for Sparse Flow Reconstruction

This repository contains a **MATLAB** script that implements and compares two algorithms for fluid flow reconstruction from sparse sensor data:  

- **Laplacian Regularized Motion Tomography (RMT)** [1]  
- **Occupation Kernel Motion Tomography (OKMT)** [2]  

The experiment demonstrates that for reconstructing a localized flow field from a very small number of sensor trajectories, the **Occupation Kernel method is both more accurate and computationally faster**.

---
## References

[1] Meriam, O., Mengxue, H., & Fumin, Z. (2024). *Laplacian regularized motion tomography for underwater vehicle flow mapping with sporadic localization measurements*. **Autonomous Robots, 48**(10).  

[2] Russo, B. P., Kamalapurkar, R., Chang, D., & Rosenfeld, J. A. (2021). *Motion Tomography via Occupation Kernels*. **arXiv preprint** arXiv:2101.02677.  

---
## Results Summary

The simulation results clearly show that **OKMT outperforms RMT** in this sparse sensing scenario.

- **Higher Accuracy**  
  - OKMT provides a much better reconstruction of the localized flow.  
  - RMT is overly smooth and diffuse, failing to capture key features.  

- **Faster Computation**  
  - OKMT solves a small system based on the number of sensors (**15 × 15**).  
  - RMT must solve a very large system based on the grid resolution (**10,000 × 10,000**).  

---

## Figures

### Figure 1: Visual Comparison
- **Ground Truth (1)**  
- **OKMT Reconstruction (2)** – visually closer to the ground truth  
- **RMT Reconstruction (3)** – overly smooth and diffuse  
- **Error Map (4)** – RMT shows higher error across most of the domain (blue areas)  

📷 *Visual Comparison of Flow Fields*

---

### Figure 2: Quantitative Comparison
- **OKMT (blue)** – lower Mean Squared Error, faster execution time  
- **RMT (orange)** – higher error and longer runtime  

📊 *Quantitative Comparison*

---

## How It Works

- **Occupation Kernel Motion Tomography (OKMT) [2]**  
  - Mesh-free method  
  - Models flow as a weighted sum of localized Gaussian "tubes" along sensor paths  
  - Solves a **small linear system** (M × M, where M = number of sensors)  

- **Laplacian Regularized Motion Tomography (RMT) [1]**  
  - Grid-based method  
  - Estimates velocity vector at each point of a high-resolution grid  
  - Requires a smoothness regularizer (Laplacian) for sparse data  
  - Solves a **large linear system** (P × P, where P = number of grid points)  

---

## How to Run the Code

### Prerequisites
- MATLAB (R2021a or newer recommended)  
- No special toolboxes required  

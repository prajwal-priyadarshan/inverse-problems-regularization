# LLM-Guided Regularization of Pseudoinverse for Ill-Posed Signal Reconstruction

## Course
**22MAT230 – Mathematics for Computing IV**  
Amrita Vishwa Vidyapeetham, Coimbatore

---

## Team Details

**Team 7 – Batch C**

| Name | Roll Number |
|-----|------------|
| G Prajwal Priyadarshan | CB.SC.U4AIE24214 |
| Kabilan K | CB.SC.U4AIE24224 |
| Kishore B | CB.SC.U4AIE24227 |
| Rahul L S | CB.SC.U4AIE24248 |

---

## Base / Reference Papers

This project is inspired and guided by the following research works:

1. Huang, C., Wang, L., Fu, M., Lu, Z. R., & Chen, Y.  
   *A Novel Iterative Integration Regularization Method for Ill-Posed Inverse Problems*,  
   **Engineering with Computers**, 37(3), 1921–1941, 2021.

2. Ji, K., et al.  
   *An Adaptive Regularized Solution to Inverse Ill-Posed Models*,  
   **IEEE Transactions on Geoscience and Remote Sensing**, 2022.

3. Alberti, G. S., et al.  
   *Learning the Optimal Tikhonov Regularizer for Inverse Problems*,  
   **NeurIPS 2021**.

These works motivated our focus on **regularization techniques**, **SVD-based analysis**, and **iterative methods** for stabilizing ill-posed inverse problems.

---

## Project Outline

### Problem Motivation
In many real-world applications such as medical imaging, signal denoising, and image restoration, the true signal cannot be observed directly. Instead, we observe noisy and distorted measurements. Recovering the original signal from such measurements leads to **inverse problems**, which are often **ill-posed**.

Ill-posed problems are challenging because:
- Small noise in measurements can cause large reconstruction errors
- The system matrix is usually ill-conditioned
- Direct inversion or pseudoinverse methods amplify noise severely

---

### Objective
The objective of this project is to:
- Study why **direct pseudoinverse fails** for ill-posed problems
- Analyze the role of **Singular Value Decomposition (SVD)** in understanding instability
- Implement and compare **regularization techniques** to obtain stable reconstructions
- Explore **iterative regularization methods** with adaptive stopping criteria

---

### Methodology Overview
1. Generate representative 1D signals:
   - Sinusoidal signal  
   - Multi-sine signal  
   - Piecewise (discontinuous) signal  

2. Apply different forward operators:
   - Gaussian blur operator  
   - Downsampling operator  
   - Rank-deficient operator  

3. Add controlled Gaussian noise to measurements

4. Perform SVD analysis and compute condition numbers

5. Compare reconstruction methods:
   - Pseudoinverse (baseline)
   - Tikhonov regularization
   - Truncated SVD (TSVD)
   - Non-Stationary Iterated Tikhonov (NSIT)

6. Evaluate performance using:
   - Relative Error
   - Mean Squared Error (MSE)
   - Peak Signal-to-Noise Ratio (PSNR)

---

## Project Updates / Current Status

- Implemented forward models for multiple operators
- Demonstrated failure of naive pseudoinverse reconstruction
- Successfully applied Tikhonov and TSVD regularization
- Performed parameter sweeps to find optimal regularization parameters
- Implemented NSIT with Morozov stopping criterion
- Achieved ~96% error reduction compared to pseudoinverse
- Extended experiments from 1D signals to 2D image reconstruction

---

## Challenges / Issues Faced

- **Severe noise amplification** during pseudoinverse reconstruction
- Choosing optimal regularization parameters (λ for Tikhonov, k for TSVD)
- High sensitivity of solutions to small singular values
- Computational cost of repeated SVD during parameter sweeps
- Balancing stability and accuracy without over-smoothing the signal
- Understanding theoretical concepts such as Picard condition and Morozov discrepancy in practice

---

## Key Results

- Pseudoinverse reconstruction completely fails for ill-posed problems
- Tikhonov and TSVD provide stable and meaningful reconstructions
- NSIT achieves the best performance by adapting regularization over iterations
- Regularization methods are robust across different signals and operators
- SVD plays a central role in explaining instability and guiding regularization

---

## Future Plans

- Extend experiments to **higher-dimensional inverse problems**
- Handle **non-Gaussian and real-world noise models**
- Explore **Fast Non-Stationary Iterated Tikhonov (FNSIT)**
- Integrate **LLM-guided adaptive parameter selection**
- Investigate **diffusion models and learned priors** for inverse problems
- Apply methods to real datasets from imaging and signal processing

---

## Conclusion

This project demonstrates that ill-posed inverse problems cannot be solved using direct inversion techniques. Regularization is not optional—it is essential. Through SVD-based analysis and carefully designed regularization methods, stable and accurate signal reconstruction becomes possible even in the presence of severe noise and ill-conditioning.

---

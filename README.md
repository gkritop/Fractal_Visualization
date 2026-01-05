# Fractal Visualization

**A high-performance interactive exploration of the Mandelbrot and Julia sets, designed to demonstrate Just-In-Time (JIT) compilation and parallel computing in Python.**

## Overview

This project is a proof-of-concept for **high-performance Python**, demonstrating how JIT compilation via **Numba** can narrow the gap between interpreted Python code and native C/C++ performance for image-processing workloads.

The application focuses on real-time fractal rendering.

---

## Key Features

- **Parallel Kernels**  
  Escape Time Algorithm implemented with Numba’s `@njit` decorator and automatic multithreading.

- **JIT Compilation**  
  Python code is compiled at runtime to optimized LLVM machine code.

- **Accessible Visualization**  
  Perceptually uniform, colorblind-friendly gradient (Deep Blue → Cyan → White).

- **Interactive Exploration**  
  Real-time zooming, panning, and switching between Mandelbrot and Julia sets at interactive frame rates.

---

## Theoretical Background

### The Mathematical Model

The application visualizes the **Mandelbrot set** and **Julia sets**, defined by iterations of a complex quadratic polynomial.

- **Mandelbrot set**  
  z_{n+1} = z_n^2 + c, z_0 = 0 
  where (c) corresponds to the pixel coordinate in the complex plane.

- **Julia sets**  
  z_{n+1} = z_n^2 + c 
  where (c) is a fixed complex constant and (z_0) corresponds to the pixel coordinate.

A point belongs to the set if (|z_n|) remains bounded as (n \to \infty).
Numerically, this is approximated using the **Escape Time Algorithm**: iterate until

|z_n| > 2

or until a maximum number of iterations (N) is reached.

---

## Technical Implementation

### Parallel Strategy (Numba)

The computational core relies on **Numba** to exploit CPU-level parallelism.

- `parallel=True` enables multithreaded execution across available CPU cores.  
- `fastmath=True` allows aggressive floating-point optimizations (e.g. vectorization, fused operations, AVX).

---

## Optimization: Continuous Coloring

To avoid visible banding artifacts, **continuous (smooth) coloring** is used instead of discrete iteration counts.

### Discrete Coloring
Color depends only on the integer number of iterations before escape.  
This approach is fast but produces noticeable color bands, especially at high zoom levels.

### Continuous Coloring
A fractional iteration value is computed:

\nu = n + 1 - \frac{\log(\log |z_n|)}{\log 2}

This value is used to interpolate smoothly between color stops, producing visually continuous gradients. The effect becomes critical during deep zooms near the fractal boundary, where discrete coloring breaks down.

---

## Features

### High-Performance Rendering
Optimized numerical kernels exploit SIMD instructions where possible, achieving real-time rendering.

### Interactive Parameter Space
- Real-time zoom and pan  
- Instant switching between Mandelbrot and Julia modes  

### Julia Morphing
Mouse position is mapped dynamically to the complex constant \(c\), enabling continuous animation of Julia set structures.

### Auto-Zoom Mode
Automated deep zooms into complex boundary regions, useful for demonstrations and benchmarking.

### Accessibility
Color palette design prioritizes luminance contrast and avoids red–green ambiguity, ensuring usability for color-vision deficiencies.

---

## Build Instructions

### Prerequisites
- **Python** 3.8 or newer  
- **pip** (Python package manager)

### Installation

```bash
# Navigate to the project directory
cd path/to/Fractal_Visualization

# Install dependencies
pip install numpy pygame numba
```

Numba handles JIT compilation; Pygame provides windowing and input handling.

---

## Running the Application

```bash
python fractal.py
```

**Note:**  
On first execution, there may be a 1–2 second delay while Numba compiles the numerical kernels. Once compilation is complete, rendering proceeds at real-time speeds.

---

## Acknowledgements

Personal project by **Giorgos Kritopoulos** to practice numerical methods, real-time visualization, and parallel computing optimization.

**Date:** 5 January 2026

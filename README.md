# kamo 🔥

My personal journey into learning about _Kolmogorov–Arnold Networks_ ([KAN](https://github.com/KindXiaoming/pykan)) using [Mojo](https://docs.modular.com/mojo/manual/).

The following excerpt from the abstract the paper [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) provides the foundational inspiration:

> Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov- Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes (“neurons”), KANs have learnable activation functions on edges (“weights”). KANs have no linear weights at all – every weight parameter is replaced by a univariate function parametrized as a spline. We show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability.

I started this journey by porting the KAN Python implementation from  [ML without tears](https://mlwithouttears.com/2024/05/15/a-from-scratch-implementation-of-kolmogorov-arnold-networks-kan/) to Mojo. This has been an excellent introduction to the world of KANs for me. One nice aspect of this clean implementation is its flexibility; it can be instantiated as either a KAN or a classic MLP, allowing for various comparisons and experiments. For now, my focus here is on understanding the fundamental concepts, rather than on performance or implementing all aspects of KANs.

# Empowering edges

The fundamental innovation of KANs lies in their learnable activation functions on edges. The paper [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) suggests using a linear combination of B-Splines and the SiLU function. Subsequent research also recommends the use of Chebyshev polynomials. These sets of basis functions from approximation theory are not only powerful but also elegant in their mathematical beauty. 

| **B-Spline** | **Chebyshev** |
|--------------|--------------|
| <img src="imgs/bspline_silu_basis.png" width="300"/> | <img src="imgs/chebyshev_basis.png" width="300"/> |

## Changelog
  
- 2024.06.13
  - Initial Commit version 2

## License

MIT
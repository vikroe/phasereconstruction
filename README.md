# Phase Reconstruction

Comparison of different Phase reconstruction algorithms for a digital holographic microscopy platform. Real-time reimplementation of one of these algorithms will be included in the following weeks.

## Technologies

The different algorithms are implemented in Matlab.

Real-time implementation will be done in C++ CUDA for an Nvidia Jetson computing device.

## References

Currently implemented algorithms are taken from (more are yet to be included) :

_iterative.m_ - [1] L. Denis, C. Fournier, T. Fournel, and C. Ducottet, “Numerical suppression of the twin image in in-line holography of a volume of micro-objects,” _Measurement Science and Technology_, vol. 19, no. 7, p. 074 004, May 2008.

_fista.m_ - [2] F. Momey, L. Denis, T. Olivier, and C. Fournier, “From fienup’s phase retrieval techniques to regularized inversion for in-line holography: Tutorial,” _Journal of the Optical Society of America A_, vol. 36, no. 12, p. D62, Nov. 2019

_fienup.m_ - [2] F. Momey, L. Denis, T. Olivier, and C. Fournier, “From fienup’s phase retrieval techniques to regularized inversion for in-line holography: Tutorial,” _Journal of the Optical Society of America A_, vol. 36, no. 12, p. D62, Nov. 2019 (it is a reimplementation of the famous Fienup's algorithm as an inverse algorithm)

_multilayer\_fista.m_ - [3] A. Berdeu, O. Flasseur, L. Méès, L. Denis, F. Momey, T. Olivier, N. Grosjean, and C. Fournier, “Reconstruction of in-line holograms: Combining model-based and regularized inversion,” Optics Express, vol. 27, no. 10, p. 14 951, May 2019
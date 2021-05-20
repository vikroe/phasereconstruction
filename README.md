# Phase Reconstruction

Implementations of the fast iterative shrinkage-thresholding algorithm for in-line digital holography. A prototype of the implementation is done in Matlab and the final implementation is done in C++ CUDA with the help of OpenCV. The final implementation was later integrated into a the already existing code for a dielectrophoretic micromanipulation platform available at <https://github.com/aa4cc/twinbeam-setup/tree/phasereconstruction/JetsonCode>.

## Usage

### Matlab prototype

Open the folder _MatlabFiles/_ and add folders _utils_ to path.

If desired, fill in new parameters to the file _parameters.m_, otherwise just run the file _main.m_ and wait for the resulting figures.

### CUDA implementation

The author tested this implementation on Ubuntu 18.04 and Ubuntu 20.04.

CUDA capable device is needed for this implementation to run. In the folder _CUDAFiles_, first, set the CUDA architecture on line 21 of _CMakeLists.txt_, current settings are for GTX 660. If OpenCV and CUDA (tested on CUDA 10.0) are installed, it should then be possible to compile the code with CMake. Then run the binary file _phasereconstruction_.

If desired, change parameters in the file _params.json_.

## Technologies

The different algorithms are implemented in Matlab.

Real-time implementation is done in C++ CUDA with OpenCV for parsing input and output media files and displaying. 
## References

The implementation of the fast iterative shrinkage-thresholding algorithm is strongly based on the algorithm described in

F. Momey, L. Denis, T. Olivier, and C. Fournier, �From fienup�s phase retrieval techniques to regularized inversion for in-line holography: Tutorial,� _Journal of the Optical Society of America A_, vol. 36, no. 12, p. D62, Nov. 2019

## Contact

In case of any questions, please ask on <v.koropecky@protonmail.com>.
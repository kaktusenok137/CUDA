# CUDA
CUDA (Compute Unified Device Architecture) is a GPGPU (General-Purpose computing on Graphics Processing Units) technology that allows programmers to implement algorithms in a simplified C programming language that can be implemented on graphics processors of eighth-generation GeForce video cards and older. CUDA technology was developed by Nvidia. In fact, CUDA allows you to include special functions in the text of a C program. These functions are written in the simplified C programming language and run on Nvidia GPUs.

# Compiling and running

To compile the program on the command line, type:
<br><b>nvcc main.cu -o MyProgram</b>
<br>Starting the program:
<br><b>./MyProgram</b>

# CUDA Profiler

NVIDIA Visual Profiler is a graphical tool that provides the ability to profile applications running on the GPU. NVIDIA Visual Profiler supports full memory bandwidth measurement within the kernel, giving developers a better understanding of what's going on in one of the most performance-critical areas of CUDA.
The profiler is invoked by the following command: nvvp.
In a profiler, open the nvcc-compiled executable.

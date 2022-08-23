# CUDA
CUDA (Compute Unified Device Architecture) is a GPGPU (General-Purpose computing on Graphics Processing Units) technology that allows programmers to implement algorithms in a simplified C programming language that can be implemented on graphics processors of eighth-generation GeForce video cards and older. CUDA technology was developed by Nvidia. In fact, CUDA allows you to include special functions in the text of a C program. These functions are written in the simplified C programming language and run on Nvidia GPUs.

# Compiling and running

To compile the program on the command line, type:
<br><b>nvcc main.cu -o MyProgram</b>
<br>Starting the program:
<br><b>./MyProgram</b>

# Laboratory work on the subject "Parallel programming".
Assignment: Develop a program in CUDA C in accordance with the options for laboratory work given in table 1. In all tasks, it is required to create a matrix (vector) or several matrices (vectors) of the same dimension indicated in Table 1, fill them with values ​​read from a text file. Text files should be prepared in advance by filling them with random numbers. Then implement the task. At the end, output the calculation results again to a text file.

The program interface is a console application. After all calculations are done, the program prints the total running time (in seconds).

Table 1. Description of tasks

<table>
  <tr>
    <td>Task</td>
    <td>Dimension of array or matrix</td>
    <td>Data type</td>
    <td>Task description</td>
  </tr>
  <tr>
    <td>Task 1</td>
    <td>[10*10; 3000*3000]</td>
    <td>double</td>
    <td>Compute the product of dense matrices C = A*B*F</td>
  </tr>
    <tr>
    <td>Task 2</td>
    <td>[10*10; 10000*10000]</td>
    <td>float</td>
    <td>Calculate value c=tr(L *U) <br>tr - matrix trace (sum of diagonal elements)
      <br>L - lower triangular matrix
<br>Y - upper triangular matrix</td>
  </tr>
  </table>

<h3>Tasl 1</h3>
To do the job, 4 files were first created:

• A.txt for matrix A;

• B.txt for matrix B;

• F.txt for matrix F;

• res.txt to record the result of the multiplication.

At runtime, you need to enter the dimension of the matrix.

The algorithm consists of multiplying matrix A by matrix B, the result is written into matrix D. After that, I multiply the resulting matrix D and matrix F, the result into matrix C.

File locations:
<ul>
  <li>lab2 - shared memory usage</li>
  <li>lab3 - CUBLAS Library (Task 1)</li>
  <li>lab3_1 - CUBLAS Library (Task 2)</li>
  </ul>

# Теория

Массивы в cuBLAS хранятся по столбцам, а не по строкам, как принято в C и C++. 
При этом сохраняется индексирование с 1. 
Чтобы упростить написание программ в рамках тех соглашений, которые приняты в языках C и C++, достаточно ввести макрос для вычисления элемента двумерного массива, 
хранимого в одномерном виде и в FORTRAN-овской нотации:

<p><i>#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) </i></p>

Здесь ld – это размерности матрицы, и в случае хранения в столбцах, является количеством строк. 
Для кода, изначально написанного на С и С++, можно было бы использовать индексирование с 0, в этом случае макрос выглядит так:
<p><i>#define IDX2C(i,j,ld) (((j)*(ld))+(i))</i></p>

Все функции для математических операций имеют следующий вид: 
<p><i>cublasStatus_t cublas(cublasHandle_t handle, ...);</i></p>

![image](https://user-images.githubusercontent.com/76211121/186202066-ba5c142a-7879-4071-98e1-3774a6c7a5d2.png)

![image](https://user-images.githubusercontent.com/76211121/186202098-cb8b100c-7369-4793-b2be-ec75b8566098.png)

<hr>

# Тестирование


![image](https://user-images.githubusercontent.com/76211121/186202542-21fb7478-2703-4364-a2a7-b6bba0619c78.png)
<p align=center>Рисунок 1 – Время умножения 3-х матриц размерностью 500х500</p>
 
 ![image](https://user-images.githubusercontent.com/76211121/186202579-6c6271b0-5b7a-4cd6-94fc-60365860a26c.png)
<p align=center>Рисунок 2 – Время умножения 3-х матриц размерностью 1000х1000</p>
 
 ![image](https://user-images.githubusercontent.com/76211121/186202603-7c7c20be-e24b-4fd9-a6d1-f35b9c8342b2.png)
<p align=center>Рисунок 3 – Время умножения 3-х матриц размерностью 1500х1500</p>
 
 ![image](https://user-images.githubusercontent.com/76211121/186202643-5a3d7ca9-bee2-45b1-b59d-da78c724f91f.png)
<p align=center>Рисунок 4 – Время умножения 3-х матриц размерностью 2000х2000</p>
 
 ![image](https://user-images.githubusercontent.com/76211121/186202681-3f6a6cc3-791b-45c7-8bca-6be9b9e7a34e.png)
<p align=center>Рисунок 5 – Время умножения 3-х матриц размерностью 2500х2500</p>
 
 ![image](https://user-images.githubusercontent.com/76211121/186202712-e453aa15-5321-4063-8842-810c251f4fad.png)
<p align=center>Рисунок 6 – Время умножения 3-х матриц размерностью 3000х3000</p>

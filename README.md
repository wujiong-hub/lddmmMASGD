# lddmmMASGD
An efficient approach for large deformation diffeomorphic metric mapping (LDDMM) for brain images by utilizing GPU-based parallel computing and a mixture automatic step size estimation method for gradient descent (MAS-GD)


Software Dependencies (with version numbers)
--------------------------------------------------
The main dependency External libraries: 
1. Insight Segmentation and Registration Toolkit (ITK) -- 5.0
2. CUDA Toolkit 8.0


Sytem Requirements
-------------------------------------
Ubuntu Xenial 16.04.3 LTS (with 8 GB RAM and NVIDIA GPU).

The lddmmMASGD is recommanded run after the histogram matching and affine registration.

Usage:
--------------------------
After the configration of system and dependencies, you can run it in the terminal:  

xxxxx$./lddmmMASGD    
Usage:  
<pre><code>./lddmmMASGD --in InputPath --ref ReferencePath --out OutputPath  
field OutputDisplacementFieldPath  
		  --scale Scale  
		  --alpha RegistrationSmoothness  
		  --sigma Sigma  
		  --epsilon LearningRate  
		  --fraction MinimumInitialEnergyFraction  
		  --steps NumberOfTimesteps  
		  --iterations MaximumNumberOfIterations  
		  --cost CostFunction  
			 0 = Mean Square Error  
			 1 = Cross Correlation  
			 2 = Mutual Information  
		  --bins Number of bins used with MI  
		  --radius The radius used with CC  
		  --MASGD  use MAS-GD optimisation  
		  --verbose ]  </code></pre>

Example:
--------------------------------
You can run the Example.py in your terminal to browse the registration performance.  

```python 
tar -zxvf ./data/template.tar.gz -C ./data/  
```
```python
tar -zxvf ./dta/target.tar.gz -C ./data/
```
```python
python Example.py
```
template image:  
![image](https://github.com/KwuJohn/lddmmMASGD/blob/master/Images/template.png)  

target image:  
![image](https://github.com/KwuJohn/lddmmMASGD/blob/master/Images/target.png)  
<pre><code>
Affine alignment  
Step translation:  
0.       -0.384558372825  
1.       -0.386345537773  
2.       -0.388387694152  
3.       -0.390462051621  
4.       -0.392321468663  
5.       -0.39391631794  
6.       -0.395304633622  
7.       -0.397748428877  
8.       -0.398856773288  
9.       -0.399841539567  
10.      -0.401661874774  
11.      -0.402397092428  
12.      -0.402994217164  
13.      -0.404365316576  
14.      -0.405657980938  
15.      -0.406811556307  
16.      -0.407812825427  
17.      -0.408665009671  
18.      -0.410648474074    
19.      -0.411215053911    
20.      -0.411849501816    
21.      -0.412866174474    
22.      -0.413751593448    
23.      -0.414479806531    
24.      -0.415138123533    
25.      -0.416626347465    
26.      -0.419190921216    
27.      -0.420346567219    
28.      -0.421322445741  
29.      -0.422128064375  
30.      -0.425139458146  
31.      -0.425630206767  
32.      -0.425962802501  
33.      -0.426137971203  
34.      -0.426156934549  
35.      -0.426017814921  
36.      -0.426267684126  
37.      -0.427015569581  
38.      -0.427626991519  
39.      -0.42807588617  
40.      -0.428351459703  
41.      -0.429812561316  
42.      -0.42992275801  
43.      -0.430382356745  
44.      -0.430706770967  
45.      -0.430836224881  
46.      -0.430765650211  
47.      -0.430782496262  
48.      -0.431463567446  
49.      -0.432026087057  
Step rigid:  
0.       -0.423585040605  
1.       -0.256156578033  
2.       -0.351133884475  
3.       -0.425361075114  
4.       -0.427248846758  
5.       -0.434268125613  
6.       -0.433261895049  
7.       -0.434525766301  
8.       -0.434908206806  
Step affine:  
0.       -0.425938664579  
1.       -0.371788200862  
2.       -0.449008335325  
3.       -0.412399252804  
4.       -0.457776317994  
5.       -0.455499764811  
6.       -0.467693127802  
7.       -0.461663463967  
8.       -0.469023801906  
9.       -0.470752774835  
10.      -0.471047073477  
11.      -0.471707720229  
Deformable alignment  
  
Step 0: alpha=0.01, scale=0.25, iteration=1000  
  
         E,     E_velocity,     E_image (E_image %),            Step_size  
0.       1.97723e+09, 0, 1.97723e+09(100%),      5.000000e-02   BLS-GD step  
1.       1.97648e+09, 13.5802, 1.97648e+09(99.9621%),    1.375000e-02   BLS-GD step  
2.       1.62451e+09, 1.05251e+07, 1.61399e+09(81.6289%),        1.512500e-02   BLS-GD step  
3.       1.51639e+09, 2.4466e+07, 1.49193e+09(75.4555%),         1.663750e-02   BLS-GD step  
4.       1.4478e+09, 4.05075e+07, 1.40729e+09(71.1749%),         1.830125e-02   BLS-GD step  
5.       1.40686e+09, 5.73714e+07, 1.34949e+09(68.2514%),        2.013138e-02   BLS-GD step  
6.       1.38144e+09, 7.47935e+07, 1.30665e+09(66.0848%),        2.214451e-02   BLS-GD step  
7.       1.36446e+09, 9.31988e+07, 1.27126e+09(64.2952%),        2.435896e-02   BLS-GD step  
8.       1.35245e+09, 1.13152e+08, 1.2393e+09(62.6785%),         2.679486e-02   BLS-GD step  
9.       1.34537e+09, 1.34543e+08, 1.21083e+09(61.2386%),        2.947435e-02   BLS-GD step  
10.      1.34195e+09, 1.6179e+08, 1.18016e+09 (59.6874%),        5.615935e-03.  MFAS-GD step  
11.      1.34879e+09, 1.79522e+08, 1.16927e+09 (59.1368%),       1.148656e-02.  MFAS-GD step  
12.      1.34195e+09, 1.6179e+08, 1.18016e+09(59.6874%),         1.148656e-02   BLS-GD step  
13.      1.34164e+09, 1.70057e+08, 1.17159e+09 (59.2539%),       1.094009e-02.  MFAS-GD step  
14.      1.34441e+09, 1.92937e+08, 1.15147e+09 (58.2366%),       1.417837e-02.  MFAS-GD step  
15.      1.34164e+09, 1.70057e+08, 1.17159e+09(59.2539%),        1.417837e-02   BLS-GD step  
16.      1.34192e+09, 1.80736e+08, 1.16118e+09 (58.7278%),       1.418700e-02.  MFAS-GD step  
17.      1.34164e+09, 1.69829e+08, 1.17181e+09(59.2652%),        1.418700e-02   BLS-GD step  
E = 1.34164e+09 (59.2728%)  
Length = 8164.9  
Time = 3.20525s (0.0534208m)  
  
Step 1: alpha=0.005, scale=0.5, iteration=1000  
  
         E,     E_velocity,     E_image (E_image %),            Step_size  
0.       1.21827e+09, 0, 1.21827e+09(100%),      5.000000e-02   BLS-GD step  
1.       1.09047e+09, 1.59183e+07, 1.07455e+09(88.2028%),        5.500000e-02   BLS-GD step  
2.       1.09033e+09, 1.59195e+07, 1.07441e+09(88.1918%),        1.512500e-02   BLS-GD step  
3.       1.07549e+09, 1.99834e+07, 1.0555e+09(86.6397%),         1.663750e-02   BLS-GD step  
4.       1.07548e+09, 1.99835e+07, 1.0555e+09(86.6393%),         4.575313e-03   BLS-GD step  
5.       1.04933e+09, 2.11921e+07, 1.02814e+09(84.3938%),        5.032844e-03   BLS-GD step  
6.       1.04297e+09, 2.30731e+07, 1.0199e+09(83.7171%),         5.536128e-03   BLS-GD step  
7.       1.03822e+09, 2.52658e+07, 1.01296e+09(83.1474%),        6.089741e-03   BLS-GD step  
8.       1.0335e+09, 2.7776e+07, 1.00572e+09(82.5536%),          6.698715e-03   BLS-GD step  
9.       1.02874e+09, 3.06456e+07, 9.98092e+08(81.9271%),        7.368587e-03   BLS-GD step  
10.      1.01908e+09, 3.7722e+07, 9.81359e+08 (80.5536%),        8.258222e-03.  MFAS-GD step  
11.      1.00995e+09, 4.66636e+07, 9.63285e+08 (79.07%),         8.917898e-03.  MFAS-GD step  
12.      9.98967e+08, 6.30378e+07, 9.35929e+08 (76.8245%),       9.616670e-03.  MFAS-GD step  
13.      9.91314e+08, 9.18607e+07, 8.99453e+08 (73.8304%),       1.091800e-02.  MFAS-GD step  
14.      9.9713e+08, 1.41258e+08, 8.55873e+08 (70.2532%),        1.272189e-02.  MFAS-GD step  
15.      9.91314e+08, 9.18607e+07, 8.99453e+08(73.8304%),        1.272189e-02   BLS-GD step  
16.      9.89993e+08, 9.81686e+07, 8.91825e+08 (73.2043%),       1.211616e-02.  MFAS-GD step  
17.      9.88621e+08, 1.13054e+08, 8.75567e+08 (71.8698%),       1.349201e-02.  MFAS-GD step  
18.      9.90171e+08, 1.38526e+08, 8.51645e+08 (69.9062%),       1.410789e-02.  MFAS-GD step  
19.      9.88621e+08, 1.13054e+08, 8.75567e+08(71.8698%),        1.410789e-02   BLS-GD step  
20.      9.88527e+08, 1.20301e+08, 8.68226e+08 (71.2672%),       1.343614e-02.  MFAS-GD step  
21.      9.89559e+08, 1.36863e+08, 8.52697e+08 (69.9925%),       1.464513e-02.  MFAS-GD step  
22.      9.88527e+08, 1.20301e+08, 8.68226e+08(71.2672%),        1.464513e-02   BLS-GD step  
23.      9.88773e+08, 1.28068e+08, 8.60705e+08 (70.6499%),       1.463129e-02.  MFAS-GD step  
24.      9.88525e+08, 1.20128e+08, 8.68398e+08(71.2813%),        1.463129e-02   BLS-GD step  
E = 9.88525e+08 (71.2906%)  
Length = 3880.78  
Time = 10.7905s (0.179842m)  
  
Step 2: alpha=0.002, scale=1.0, iteration=1000  
  
         E,     E_velocity,     E_image (E_image %),            Step_size  
0.       9.17556e+08, 0, 9.17556e+08(100%),      5.000000e-02   BLS-GD step  
1.       7.57234e+08, 2.07086e+07, 7.36526e+08(80.2704%),        5.500000e-02   BLS-GD step  
2.       7.57095e+08, 2.07125e+07, 7.36382e+08(80.2548%),        1.512500e-02   BLS-GD step  
3.       7.57092e+08, 2.07126e+07, 7.36379e+08(80.2544%),        4.159375e-03   BLS-GD step  
4.       7.40055e+08, 2.18881e+07, 7.18167e+08(78.2696%),        4.575313e-03   BLS-GD step  
5.       7.32001e+08, 2.35417e+07, 7.08459e+08(77.2116%),        5.032844e-03   BLS-GD step  
6.       7.24947e+08, 2.55472e+07, 6.994e+08(76.2242%),          5.536128e-03   BLS-GD step  
7.       7.18316e+08, 2.79164e+07, 6.904e+08(75.2433%),          6.089741e-03   BLS-GD step  
8.       7.11835e+08, 3.06792e+07, 6.81156e+08(74.2359%),        6.698715e-03   BLS-GD step  
9.       7.05751e+08, 3.38715e+07, 6.71879e+08(73.2249%),        7.368587e-03   BLS-GD step  
10.      6.98647e+08, 3.80418e+07, 6.60605e+08 (71.9961%),       1.016029e-03.  MFAS-GD step  
11.      6.95856e+08, 3.96116e+07, 6.56244e+08 (71.5209%),       1.545369e-03.  MFAS-GD step  
12.      6.93732e+08, 4.37244e+07, 6.50008e+08 (70.8412%),       2.560698e-03.  MFAS-GD step  
13.      6.91683e+08, 4.51973e+07, 6.46485e+08 (70.4573%),       6.725055e-04.  MFAS-GD step  
14.      6.87818e+08, 4.71181e+07, 6.407e+08 (69.8268%),         6.954186e-04.  MFAS-GD step  
15.      6.82591e+08, 5.04008e+07, 6.32191e+08 (68.8994%),       9.614383e-04.  MFAS-GD step  
16.      6.96783e+08, 7.0048e+07, 6.26735e+08 (68.3049%),        4.316934e-03.  MFAS-GD step  
17.      6.82591e+08, 5.04008e+07, 6.32191e+08(68.8994%),        4.316934e-03   BLS-GD step  
18.      6.80372e+08, 5.26618e+07, 6.27711e+08 (68.4112%),       4.111424e-03.  MFAS-GD step  
19.      6.75923e+08, 5.7839e+07, 6.18084e+08 (67.362%),         4.546172e-03.  MFAS-GD step  
20.      6.70441e+08, 6.57495e+07, 6.04691e+08 (65.9024%),       4.392612e-03.  MFAS-GD step  
21.      6.64239e+08, 7.87151e+07, 5.85524e+08 (63.8135%),       5.027336e-03.  MFAS-GD step  
22.      6.59994e+08, 9.51657e+07, 5.64828e+08 (61.5579%),       4.720593e-03.  MFAS-GD step  
23.      6.59669e+08, 1.22677e+08, 5.36992e+08 (58.5241%),       6.010714e-03.  MFAS-GD step  
24.      6.65136e+08, 1.50604e+08, 5.14532e+08 (56.0764%),       4.841047e-03.  MFAS-GD step  
25.      6.59669e+08, 1.22677e+08, 5.36992e+08(58.5241%),        4.841047e-03   BLS-GD step  
26.      6.59476e+08, 1.251e+08, 5.34377e+08 (58.2391%),         4.825236e-03.  MFAS-GD step  
27.      6.59322e+08, 1.30698e+08, 5.28625e+08 (57.6123%),       5.304851e-03.  MFAS-GD step  
28.      6.59519e+08, 1.38264e+08, 5.21255e+08 (56.8091%),       4.637768e-03.  MFAS-GD step  
29.      6.59322e+08, 1.30698e+08, 5.28625e+08(57.6123%),        4.637768e-03   BLS-GD step  
30.      6.59314e+08, 1.33008e+08, 5.26307e+08 (57.3596%),       4.416944e-03.  MFAS-GD step  
31.      6.59613e+08, 1.38893e+08, 5.2072e+08 (56.7508%),        5.521353e-03.  MFAS-GD step  
32.      6.59314e+08, 1.33008e+08, 5.26307e+08(57.3596%),        5.521353e-03   BLS-GD step  
33.      6.59409e+08, 1.35878e+08, 5.23531e+08 (57.0571%),       5.514098e-03.  MFAS-GD step  
34.      6.59314e+08, 1.32984e+08, 5.2633e+08(57.3622%),         5.514098e-03   BLS-GD step  
E = 6.59314e+08 (57.3622%)  
Length = 3851.05  
Time = 108.226s (1.80376m)   </code></pre>
transformed template image:  
  
 


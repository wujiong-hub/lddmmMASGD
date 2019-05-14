# lddmmMASGD
An efficient approach for large deformation diffeomorphic metric mapping (LDDMM) for brain images by utilizing GPU-based parallel computing and a mixture automatic step size estimation method for gradient descent (MAS-GD)


Software Dependencies (with version numbers)
--------------------------------------------------
The main dependency External libraries: 
1. Insight Segmentation and Registration Toolkit (ITK) -- 5.0
2. SimpleITK 
3. CUDA Toolkit 10.0  


Sytem Requirements
-------------------------------------
Ubuntu Xenial 16.04.3 LTS (with 8 GB RAM and NVIDIA GPU with not less than 2 GB global memory).

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
chmod -R 775 ./Bin/
```
```python
python Example.py
```
template image:  
![image](https://github.com/KwuJohn/lddmmMASGD/blob/master/Images/template.png)  

target image:  
![image](https://github.com/KwuJohn/lddmmMASGD/blob/master/Images/target.png)  
<pre><code>
===================================================
*************** Do the preprocessing **************
===================================================



===================================================
*********** Do the Affine Registration ************
===================================================



Number of input arguments = 
        6
Parameters = 
        0 = /mnt/postData/LDDMM_experiments/version3/Bin/
        1 = ./result/step0_inite/template.img
        2 = ./result/step0_inite/target.img
        3 = result/step1_affine/template_to_target/
        4 = 1
        5 = template_to_target_affined.img

*************************************************
*********************STEP1***********************
*************************************************

0) Template   = 
        template img = ./result/step0_inite/template.img                 FILE EXISTS
        template hdr = ./result/step0_inite/template.hdr                 FILE EXISTS
1) Target   = 
        target img   = ./result/step0_inite/target.img           FILE EXISTS
        target hdr   = ./result/step0_inite/target.hdr           FILE EXISTS
2) Output folder   = 
        output path  = result/step1_affine/template_to_target/           DIRECTORY EXISTS
3) Do histogram matching before AIR   = 
        YES
4) Output   = 
        output img = template_to_target_affined.img 

*************************************************
*********************STEP2***********************
*************************************************

CREATING OUTPUT TEMPORARY DIRECTORY AND COPYING FILES

*************************************************
*********************STEP3***********************
*************************************************

GLOBAL HISTOGRAM MATCHING


INPUTS TO THE PROGRAM
0> /mnt/postData/LDDMM_experiments/version3/Bin//IMG_histmatch4
1> template_to_target.tmp/template.img
2> template_to_target.tmp/target.img
3> template_to_target.tmp/template_h.img
4> template_to_target.tmp/target_h.img
5> 1024
6> 3
7> 0
8> 1
9> template_to_target.tmp/template.hist
10> template_to_target.tmp/target.hist
11> template_to_target.tmp/template_h.hist
12> template_to_target.tmp/target_h.hist


inputfile1              template_to_target.tmp/template.img
inputfile2              template_to_target.tmp/target.img
outputfile1             template_to_target.tmp/template_h.img
outputfile2             template_to_target.tmp/target_h.img
Nbins                   1024
sigma                   3
matchendpoints          0
matchhistogram          1
inputfile1_hist         template_to_target.tmp/template.hist
inputfile2_hist         template_to_target.tmp/target.hist
outputfile1_hist        template_to_target.tmp/template_h.hist
outputfile2_hist        template_to_target.tmp/target_h.hist


*****READING IMAGES AND ALLOCATING ARRAYS*****
Image-1 max-min values:         255             0
Image-2 max-min values:         255             0
Error between histograms=       0.000117029
Image-1 max-min values:         255             0
Image-2 max-min values:         255             0
Error between histograms=       0.000117029
*****MATCHING HISTOGRAMS*****
Image-1 max-min values:         255             0
Image-2 max-min values:         255             0
Error between histograms=       7.49732e-06
*****IMAGES ARE BEING SAVED*****

 Inside AnalyzeImage::saveAnalyzeImageData() 
 File template_to_target.tmp/template_h.img opened successfully 
 size_of_data_in_bytes = 31464000
 Inside AnalyzeImage::saveAnalyzeImageData() 
 File template_to_target.tmp/target_h.img opened successfully 
 size_of_data_in_bytes = 31464000
*****TOTAL RUNNING TIME*****
6.80552



*************************************************
*********************STEP4***********************
*************************************************

AIR CALCULATIONS


INPUTS TO THE PROGRAM
0> /mnt/postData/LDDMM_experiments/version3/Bin//IMG_apply_AIR_tform
1> template_to_target.tmp/template.img
2> template_to_target.tmp/template_d.img
3> template_to_target.tmp/final_air.txt
4> 1
5> template_to_target.tmp/template.imgsize
6> 1

190     230     180
190     230     180

0.989353        -0.000350868    0.0194772       -0.450871       
0.0158836       1.03171 -0.0821415      -0.230368       
0.0249017       0.0896851       0.928384        -0.754377       
0       0       0       1       


1.01126 0.00217147      -0.0210237      0.440586        
-0.0175929      0.961826        0.0854695       0.278118        
-0.025425       -0.092974       1.06945 0.773885        
0       0       0       1       

190     230     180     1       1       1       

 Inside AnalyzeImage::saveAnalyzeImageData() 
 File template_to_target.tmp/template_d.img opened successfully 
 size_of_data_in_bytes = 31464000

*************************************************
*********************STEP5***********************
*************************************************

CLEANING TEMPORARY FILES

FINISHED...


===================================================
*********** Do the Histogram Matching *************
===================================================


result/step1_affine/template_to_target/template_to_target_affined.img
./result/step0_inite/target.img

INPUTS TO THE PROGRAM
0> /mnt/postData/LDDMM_experiments/version3/Bin/IMG_histmatch4
1> result/step1_affine/template_to_target/template_to_target_affined.img
2> ./result/step0_inite/target.img
3> result/step2_hist//template_to_target_affined_to_target/template_to_target_affined_to_target_histMatched.img
4> result/step2_hist//template_to_target_affined_to_target/target_histMatched.img
5> 1024
6> 3
7> 0
8> 1
9> result/step2_hist//template_to_target_affined_to_target/template_to_target_affined_to_target_hist1
10> result/step2_hist//template_to_target_affined_to_target/target_hist1
11> result/step2_hist//template_to_target_affined_to_target/template_to_target_affined_to_target_hist2
12> result/step2_hist//template_to_target_affined_to_target/target_hist2


inputfile1              result/step1_affine/template_to_target/template_to_target_affined.img
inputfile2              ./result/step0_inite/target.img
outputfile1             result/step2_hist//template_to_target_affined_to_target/template_to_target_affined_to_target_histMatched.img
outputfile2             result/step2_hist//template_to_target_affined_to_target/target_histMatched.img
Nbins                   1024
sigma                   3
matchendpoints          0
matchhistogram          1
inputfile1_hist         result/step2_hist//template_to_target_affined_to_target/template_to_target_affined_to_target_hist1
inputfile2_hist         result/step2_hist//template_to_target_affined_to_target/target_hist1
outputfile1_hist        result/step2_hist//template_to_target_affined_to_target/template_to_target_affined_to_target_hist2
outputfile2_hist        result/step2_hist//template_to_target_affined_to_target/target_hist2


*****READING IMAGES AND ALLOCATING ARRAYS*****
Image-1 max-min values:         254.751         0
Image-2 max-min values:         255             0
Error between histograms=       0.00546247
Image-1 max-min values:         254.751         0
Image-2 max-min values:         255             0
Error between histograms=       0.00546247
*****MATCHING HISTOGRAMS*****
Image-1 max-min values:         255             0
Image-2 max-min values:         255             0
Error between histograms=       7.37313e-06
*****IMAGES ARE BEING SAVED*****

 Inside AnalyzeImage::saveAnalyzeImageData() 
 File result/step2_hist//template_to_target_affined_to_target/template_to_target_affined_to_target_histMatched.img opened successfully 
 size_of_data_in_bytes = 31464000
 Inside AnalyzeImage::saveAnalyzeImageData() 
 File result/step2_hist//template_to_target_affined_to_target/target_histMatched.img opened successfully 
 size_of_data_in_bytes = 31464000
*****TOTAL RUNNING TIME*****
6.64489



===================================================
*********** Do the lddmmMASGD Registration ********
===================================================

1> Template image:  result/step2_hist//template_to_target_affined_to_target/template_to_target_affined_to_target_histMatched.img
2> Target img:  result/step2_hist//template_to_target_affined_to_target/target_histMatched.img
3> Metric: Cross Correlation


Step 0: alpha=0.01, scale=0.25, iteration=1000

         E,     E_velocity,     E_image (E_image %),            Step_size
0.       -3.52181e+12, 0, -3.52181e+12(100%),    5.000000e-02   BLS-GD step
1.       -3.56526e+12, 3.55675e+06, -3.56526e+12(92.4586%),      5.500000e-02   BLS-GD step
2.       -3.58724e+12, 8.32238e+06, -3.58725e+12(88.6433%),      6.050000e-02   BLS-GD step
3.       -3.60612e+12, 1.63938e+07, -3.60614e+12(85.3651%),      6.655000e-02   BLS-GD step
4.       -3.62055e+12, 2.51855e+07, -3.62057e+12(82.8603%),      7.320500e-02   BLS-GD step
5.       -3.63314e+12, 3.68644e+07, -3.63318e+12(80.6733%),      8.052550e-02   BLS-GD step
6.       -3.63997e+12, 4.65256e+07, -3.64002e+12(79.4857%),      8.857805e-02   BLS-GD step
7.       -3.64001e+12, 4.65368e+07, -3.64006e+12(79.4794%),      2.435896e-02   BLS-GD step
8.       -3.64669e+12, 5.03424e+07, -3.64674e+12(78.3189%),      2.679486e-02   BLS-GD step
9.       -3.64971e+12, 5.40116e+07, -3.64976e+12(77.7952%),      2.947435e-02   BLS-GD step
10.      -3.6553e+12, 6.56513e+07, -3.65537e+12 (76.822%),       6.092153e-02.  MAS-GD step
11.      -3.66663e+12, 8.53932e+07, -3.66672e+12 (74.8522%),     7.020104e-02.  MAS-GD step
12.      -3.68317e+12, 1.25686e+08, -3.68329e+12 (71.9762%),     8.674193e-02.  MAS-GD step
13.      -3.69852e+12, 1.81172e+08, -3.6987e+12 (69.3021%),      8.739609e-02.  MAS-GD step
14.      -3.70814e+12, 2.22406e+08, -3.70836e+12 (67.6261%),     5.851865e-02.  MAS-GD step
15.      -3.71608e+12, 2.60708e+08, -3.71634e+12 (66.2411%),     5.629652e-02.  MAS-GD step
16.      -3.72219e+12, 2.95709e+08, -3.72249e+12 (65.1743%),     5.798493e-02.  MAS-GD step
17.      -3.72602e+12, 3.25836e+08, -3.72635e+12 (64.5042%),     6.103032e-02.  MAS-GD step
18.      -3.72727e+12, 3.52477e+08, -3.72763e+12 (64.2825%),     7.189995e-02.  MAS-GD step
19.      -3.72624e+12, 3.66736e+08, -3.7266e+12 (64.4602%),      6.175223e-02.  MAS-GD step
20.      -3.72727e+12, 3.52477e+08, -3.72763e+12(64.2825%),      6.175223e-02   BLS-GD step
21.      -3.72698e+12, 3.38724e+08, -3.72732e+12 (64.3363%),     6.015211e-02.  MAS-GD step
22.      -3.72752e+12, 3.52563e+08, -3.72788e+12(64.2392%),      6.015211e-02   BLS-GD step
E = -3.72755e+12 (64.2337%)
Length = 8355.71
Time = 4.04984s (0.0674973m)

Step 1: alpha=0.005, scale=0.5, iteration=1000

         E,     E_velocity,     E_image (E_image %),            Step_size
0.       -3.61264e+12, 0, -3.61264e+12(100%),    5.000000e-02   BLS-GD step
1.       -3.63091e+12, 1.57022e+06, -3.63092e+12(95.855%),       5.500000e-02   BLS-GD step
2.       -3.63799e+12, 4.18534e+06, -3.63799e+12(94.2498%),      6.050000e-02   BLS-GD step
3.       -3.638e+12, 4.1863e+06, -3.63801e+12(94.2469%),         1.663750e-02   BLS-GD step
4.       -3.64263e+12, 5.06065e+06, -3.64263e+12(93.1975%),      1.830125e-02   BLS-GD step
5.       -3.64494e+12, 6.16682e+06, -3.64495e+12(92.672%),       2.013138e-02   BLS-GD step
6.       -3.64732e+12, 7.46903e+06, -3.64732e+12(92.1331%),      2.214451e-02   BLS-GD step
7.       -3.64972e+12, 8.97355e+06, -3.64973e+12(91.5873%),      2.435896e-02   BLS-GD step
8.       -3.65211e+12, 1.07215e+07, -3.65212e+12(91.0457%),      2.679486e-02   BLS-GD step
9.       -3.65458e+12, 1.27292e+07, -3.65459e+12(90.4843%),      2.947435e-02   BLS-GD step
10.      -3.65879e+12, 1.67405e+07, -3.6588e+12 (89.5293%),      2.146586e-02.  MAS-GD step
11.      -3.66286e+12, 2.14945e+07, -3.66288e+12 (88.6046%),     2.700389e-02.  MAS-GD step
12.      -3.66723e+12, 2.74272e+07, -3.66726e+12 (87.6111%),     2.021786e-02.  MAS-GD step
13.      -3.67669e+12, 4.44222e+07, -3.67674e+12 (85.4611%),     3.694235e-02.  MAS-GD step
14.      -3.68039e+12, 5.22512e+07, -3.68044e+12 (84.6214%),     1.253263e-02.  MAS-GD step
15.      -3.6918e+12, 8.90815e+07, -3.69189e+12 (82.0241%),      4.345689e-02.  MAS-GD step
16.      -3.69354e+12, 9.27946e+07, -3.69364e+12 (81.6283%),     3.724509e-03.  MAS-GD step
17.      -3.696e+12, 9.77749e+07, -3.69609e+12 (81.0705%),       4.776377e-03.  MAS-GD step
18.      -3.69262e+12, 1.2211e+08, -3.69274e+12 (81.8314%),      2.060038e-02.  MAS-GD step
19.      -3.696e+12, 9.77749e+07, -3.69609e+12(81.0705%),        2.060038e-02   BLS-GD step
20.      -3.6964e+12, 9.80124e+07, -3.6965e+12 (80.9786%),       1.963629e-02.  MAS-GD step
21.      -3.69726e+12, 9.89277e+07, -3.69736e+12 (80.7846%),     2.866156e-02.  MAS-GD step
22.      -3.69796e+12, 9.99866e+07, -3.69806e+12 (80.6258%),     1.614905e-02.  MAS-GD step
23.      -3.69951e+12, 1.07943e+08, -3.69961e+12 (80.2725%),     6.242965e-02.  MAS-GD step
24.      -3.70033e+12, 1.08769e+08, -3.70044e+12 (80.085%),      4.173670e-03.  MAS-GD step
25.      -3.70145e+12, 1.10355e+08, -3.70156e+12 (79.8298%),     6.171378e-03.  MAS-GD step
26.      -3.69393e+12, 1.19209e+08, -3.69405e+12 (81.5346%),     2.525719e-02.  MAS-GD step
27.      -3.70145e+12, 1.10355e+08, -3.70156e+12(79.8298%),      2.525719e-02   BLS-GD step
28.      -3.70182e+12, 1.11596e+08, -3.70194e+12 (79.7457%),     2.426502e-02.  MAS-GD step
29.      -3.7027e+12, 1.14355e+08, -3.70282e+12 (79.5457%),      2.690820e-02.  MAS-GD step
30.      -3.70412e+12, 1.1985e+08, -3.70424e+12 (79.2231%),      3.611455e-02.  MAS-GD step
31.      -3.70489e+12, 1.22738e+08, -3.70501e+12 (79.0488%),     1.541742e-02.  MAS-GD step
32.      -3.70646e+12, 1.33737e+08, -3.70659e+12 (78.69%),       4.935321e-02.  MAS-GD step
33.      -3.70721e+12, 1.3478e+08, -3.70735e+12 (78.5178%),      4.499963e-03.  MAS-GD step
34.      -3.70819e+12, 1.36381e+08, -3.70833e+12 (78.2958%),     6.725049e-03.  MAS-GD step
35.      -3.70378e+12, 1.42474e+08, -3.70392e+12 (79.295%),      2.307049e-02.  MAS-GD step
36.      -3.70819e+12, 1.36381e+08, -3.70833e+12(78.2958%),      2.307049e-02   BLS-GD step
37.      -3.70828e+12, 1.36383e+08, -3.70841e+12 (78.2766%),     2.305597e-02.  MAS-GD step
38.      -3.70852e+12, 1.36516e+08, -3.70866e+12 (78.2201%),     2.165943e-02.  MAS-GD step
39.      -3.70882e+12, 1.37061e+08, -3.70895e+12 (78.1535%),     2.626242e-02.  MAS-GD step
40.      -3.70903e+12, 1.37797e+08, -3.70917e+12 (78.1047%),     1.725130e-02.  MAS-GD step
41.      -3.70915e+12, 1.40013e+08, -3.70929e+12 (78.0785%),     3.140271e-02.  MAS-GD step
42.      -3.70951e+12, 1.40918e+08, -3.70966e+12 (77.9946%),     8.994750e-03.  MAS-GD step
43.      -3.70993e+12, 1.43081e+08, -3.71008e+12 (77.8989%),     1.650806e-02.  MAS-GD step
44.      -3.71026e+12, 1.44409e+08, -3.7104e+12 (77.8251%),      8.459081e-03.  MAS-GD step
45.      -3.71084e+12, 1.45927e+08, -3.71098e+12 (77.6936%),     8.423223e-03.  MAS-GD step
46.      -3.70945e+12, 1.52563e+08, -3.70961e+12 (78.0058%),     3.067871e-02.  MAS-GD step
47.      -3.71084e+12, 1.45927e+08, -3.71098e+12(77.6936%),      3.067871e-02   BLS-GD step
48.      -3.71099e+12, 1.46519e+08, -3.71114e+12 (77.6582%),     2.926567e-02.  MAS-GD step
49.      -3.71131e+12, 1.4729e+08, -3.71146e+12 (77.5863%),      1.951409e-02.  MAS-GD step
50.      -3.71162e+12, 1.51105e+08, -3.71177e+12 (77.5141%),     6.438126e-02.  MAS-GD step
51.      -3.71202e+12, 1.5141e+08, -3.71217e+12 (77.4249%),      5.314419e-03.  MAS-GD step
52.      -3.712e+12, 1.51954e+08, -3.71215e+12 (77.4284%),       9.105189e-03.  MAS-GD step
53.      -3.71202e+12, 1.5141e+08, -3.71217e+12(77.4249%),       9.105189e-03   BLS-GD step
54.      -3.7122e+12, 1.5139e+08, -3.71235e+12 (77.3837%),       8.671860e-03.  MAS-GD step
55.      -3.70805e+12, 1.51558e+08, -3.70821e+12 (78.3235%),     3.938622e-02.  MAS-GD step
56.      -3.7122e+12, 1.5139e+08, -3.71235e+12(77.3837%),        3.938622e-02   BLS-GD step
57.      -3.71215e+12, 1.51411e+08, -3.7123e+12 (77.3946%),      3.765703e-02.  MAS-GD step
58.      -3.7122e+12, 1.5139e+08, -3.71235e+12(77.3836%),        3.765703e-02   BLS-GD step
E = -3.7122e+12 (77.3836%)
Length = 4241.67
Time = 28.4629s (0.474381m)

Step 2: alpha=0.002, scale=1.0, iteration=1000

         E,     E_velocity,     E_image (E_image %),            Step_size
0.       -3.57624e+12, 0, -3.57624e+12(100%),    5.000000e-02   BLS-GD step
1.       -3.59205e+12, 2.42832e+06, -3.59206e+12(96.204%),       5.500000e-02   BLS-GD step
2.       -3.59208e+12, 2.42841e+06, -3.59208e+12(96.1984%),      1.512500e-02   BLS-GD step
3.       -3.6001e+12, 3.10456e+06, -3.6001e+12(94.2725%),        1.663750e-02   BLS-GD step
4.       -3.60316e+12, 4.45251e+06, -3.60316e+12(93.5388%),      1.830125e-02   BLS-GD step
5.       -3.60469e+12, 6.06745e+06, -3.60469e+12(93.171%),       2.013138e-02   BLS-GD step
6.       -3.60469e+12, 6.0675e+06, -3.6047e+12(93.1707%),        5.536128e-03   BLS-GD step
7.       -3.61125e+12, 6.43749e+06, -3.61126e+12(91.5955%),      6.089741e-03   BLS-GD step
8.       -3.61309e+12, 7.02214e+06, -3.61309e+12(91.1555%),      6.698715e-03   BLS-GD step
9.       -3.61455e+12, 7.69289e+06, -3.61456e+12(90.803%),       7.368587e-03   BLS-GD step
10.      -3.61812e+12, 9.48886e+06, -3.61813e+12 (89.9479%),     9.673639e-03.  MAS-GD step
11.      -3.62232e+12, 1.1938e+07, -3.62233e+12 (88.9394%),      1.048350e-02.  MAS-GD step
12.      -3.62863e+12, 1.64087e+07, -3.62864e+12 (87.4237%),     1.122152e-02.  MAS-GD step
13.      -3.6371e+12, 2.41766e+07, -3.63712e+12 (85.389%),       1.250904e-02.  MAS-GD step
14.      -3.64772e+12, 3.73116e+07, -3.64776e+12 (82.8362%),     1.429565e-02.  MAS-GD step
15.      -3.65853e+12, 5.57165e+07, -3.65858e+12 (80.2388%),     1.439002e-02.  MAS-GD step
16.      -3.66778e+12, 7.64492e+07, -3.66786e+12 (78.0135%),     1.260684e-02.  MAS-GD step
17.      -3.67598e+12, 9.90668e+07, -3.67608e+12 (76.0409%),     1.145545e-02.  MAS-GD step
18.      -3.6834e+12, 1.23002e+08, -3.68352e+12 (74.2543%),      1.060909e-02.  MAS-GD step
19.      -3.69021e+12, 1.47768e+08, -3.69036e+12 (72.6146%),     9.975802e-03.  MAS-GD step
20.      -3.69664e+12, 1.73386e+08, -3.69682e+12 (71.0639%),     9.651500e-03.  MAS-GD step
21.      -3.70281e+12, 2.00068e+08, -3.70301e+12 (69.5775%),     9.610634e-03.  MAS-GD step
22.      -3.70869e+12, 2.27366e+08, -3.70892e+12 (68.1606%),     9.574747e-03.  MAS-GD step
23.      -3.71416e+12, 2.54552e+08, -3.71441e+12 (66.8418%),     9.450408e-03.  MAS-GD step
24.      -3.71918e+12, 2.81021e+08, -3.71946e+12 (65.6306%),     9.279284e-03.  MAS-GD step
25.      -3.72378e+12, 3.06269e+08, -3.72408e+12 (64.5212%),     9.081024e-03.  MAS-GD step
26.      -3.72796e+12, 3.29839e+08, -3.72829e+12 (63.5116%),     8.847520e-03.  MAS-GD step
27.      -3.73178e+12, 3.51815e+08, -3.73213e+12 (62.5902%),     8.753602e-03.  MAS-GD step
28.      -3.73535e+12, 3.72429e+08, -3.73572e+12 (61.7282%),     8.848743e-03.  MAS-GD step
29.      -3.73855e+12, 3.91315e+08, -3.73894e+12 (60.955%),      8.874137e-03.  MAS-GD step
30.      -3.74128e+12, 4.07881e+08, -3.74169e+12 (60.2955%),     8.685333e-03.  MAS-GD step
31.      -3.74354e+12, 4.22241e+08, -3.74396e+12 (59.7499%),     8.596648e-03.  MAS-GD step
32.      -3.74536e+12, 4.34699e+08, -3.7458e+12 (59.3103%),      8.735825e-03.  MAS-GD step
33.      -3.74667e+12, 4.45035e+08, -3.74712e+12 (58.9937%),     8.762552e-03.  MAS-GD step
34.      -3.74748e+12, 4.52931e+08, -3.74793e+12 (58.7975%),     8.501041e-03.  MAS-GD step
35.      -3.74785e+12, 4.58587e+08, -3.74831e+12 (58.7071%),     8.373738e-03.  MAS-GD step
36.      -3.74783e+12, 4.62221e+08, -3.74829e+12 (58.7108%),     8.499865e-03.  MAS-GD step
37.      -3.74785e+12, 4.58587e+08, -3.74831e+12(58.7071%),      8.499865e-03   BLS-GD step
38.      -3.74873e+12, 4.5474e+08, -3.74918e+12 (58.4983%),      9.024573e-03.  MAS-GD step
39.      -3.74945e+12, 4.45956e+08, -3.7499e+12 (58.3257%),      1.087113e-02.  MAS-GD step
40.      -3.74994e+12, 4.3076e+08, -3.75037e+12 (58.2124%),      1.333661e-02.  MAS-GD step
41.      -3.74923e+12, 4.09351e+08, -3.74964e+12 (58.3876%),     1.548482e-02.  MAS-GD step
42.      -3.74994e+12, 4.3076e+08, -3.75037e+12(58.2124%),       1.548482e-02   BLS-GD step
43.      -3.74987e+12, 4.25861e+08, -3.7503e+12 (58.23%),        1.540369e-02.  MAS-GD step
44.      -3.74994e+12, 4.30973e+08, -3.75037e+12(58.2122%),      1.540369e-02   BLS-GD step
E = -3.74994e+12 (58.2121%)
Length = 5611.48
Time = 139.577s (2.32628m)

===================================================
*********** apply field to image and saving *******
===================================================

1> image:  ./data/template.nii.gz
2> save file:  result/step3_lddmm/lddmmcc/LDDMM_Images//template_to_target.nii.gz
3> field: ./result/lddmmcc/field/template_to_target_combine.vtk
</code></pre>

transformed template image:  
![image](https://github.com/KwuJohn/lddmmMASGD/blob/master/Images/transformed.png)
 


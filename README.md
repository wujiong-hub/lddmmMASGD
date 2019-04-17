# lddmmMASGD
An efficient approach for large deformation diffeomorphic metric mapping (LDDMM) for brain images by utilizing GPU-based parallel computing and a mixture automatic step size estimation method for gradient descent (MAS-GD)


Software Dependencies (with version numbers)
--------------------------------------------------
The main dependency External libraries: 
1. Insight Segmentation and Registration Toolkit (ITK) -- 5.0
2. CUDA Toolkit 8.0


Sytem Requirements
-------------------------------------
Ubuntu Xenial 16.04.3 LTS (with 64 GB RAM).

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


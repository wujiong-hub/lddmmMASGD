# -*- coding: utf-8 -*-
"""
This is a temporary script file.

This example depicts the registration pipline including histogram matching, 

affine and lddmmMASGD registration. 
"""

from Bin.Registration import *
from Bin.Evaluation import *
from Bin.util import *

inImgPath = './data/'
refImgPath = './data/'

outIniteDir = './result/step0_inite/'
outAffineDir = './result/step1_affine/'
outHistDir = './result/step2_hist/'
outlddmmDir = './result/step3_lddmm/Middle/'
outFieldDir = './result/lddmmcc/field/'
imgSaveDir = './result/step3_lddmm/lddmmcc/LDDMM_Images/'
labelSaveDir = './result/step3_lddmm/lddmmcc/LDDMM_Labels/'

dirMake(outAffineDir)
dirMake(outHistDir)
dirMake(outlddmmDir)
dirMake(outFieldDir)
dirMake(imgSaveDir)
dirMake(labelSaveDir)


inImgDir = inImgPath + 'template.nii.gz'
refImgDir = refImgPath + 'target.nii.gz'

inImg = imgRead(inImgDir)
refImg = imgRead(refImgDir)

imgShow(inImg)
imgShow(refImg)

field = imgRegistration(inImgDir, refImgDir, alphaList = [0.01,0.005,0.002], scaleList=[0.25,0.5,1.0], Metric=1, iterations=[1000,1000,1000], MASGD=True, useNearest=False, verbose=True, debug=False, outIniteDir=outIniteDir, outAffineDir=outAffineDir, outHistDir=outHistDir, outlddmmDir=outlddmmDir, outFieldDir=outFieldDir)
imgApplyFieldSave(inImgDir, refImgDir, outFieldDir, outDirPath=imgSaveDir)

transfedImg = imgRead(imgSaveDir+'template_to_target.nii.gz')
imgShow(transfedImg)


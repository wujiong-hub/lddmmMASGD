# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nibabel as ni
from BrainRegistration import *

inImgDir = './data/template.nii'
refImgDir = './data/target.nii.gz'
outlddmmDir = './lddmmResults/'
dirMake(outlddmmDir)

inImg = imgRead(inImgDir)
refImg = imgRead(refImgDir)

imgShow(inImg)
imgShow(refImg)

field = imgRegistration(inImg, refImg, scale=1.0, affineScale=1.0, lddmmScaleList=[0.25,0.5,1.0], lddmmAlphaList=[0.01,0.005,0.002], iterations=[1000,1000,1000], useMI=True, Metric=0, useNearest=True, inAffine=identityAffine, padding=0, inMask=None, refMask=None, verbose=True, outDirPath=outlddmmDir)
transfedImg = imgApplyField(inImg, field, size=refImg.GetSize(), spacing=refImg.GetSpacing())
imgWrite(transfedImg, outlddmmDir+'aligned.nii')

imgShow(transfedImg)

ImageData = ni.load(outlddmmDir+'aligned.nii').get_data()
ImageData = ImageData[::-1][::-1][:]
Image = ni.Nifti1Image(ImageData, ni.load(refImgDir).affine)
os.system('rm' + outlddmmDir + ' aligned.nii')
ni.save(Image, outlddmmDir + 'aligned.nii')

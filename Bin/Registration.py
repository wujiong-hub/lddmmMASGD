#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import os, math, sys, subprocess, tempfile, shutil
import os.path
from util import *
from preprocessor import *
import matplotlib.pyplot as plt

dimension = 3
vectorComponentType = sitk.sitkFloat32
vectorType = sitk.sitkVectorFloat32
affine = sitk.AffineTransform(dimension)
identityAffine = list(affine.GetParameters())
identityDirection = list(affine.GetMatrix())
zeroOrigin = [0]*dimension
zeroIndex = [0]*dimension
lddmmMASGDDirPath = os.path.dirname(os.path.realpath(__file__))+"/"
airBinPath = os.path.dirname(os.path.realpath(__file__))+"/"
histMatchBinPath = os.path.dirname(os.path.realpath(__file__))+"/"

def affineToField(affine, size, spacing):
    """
    Generates displacement field with given size and spacing based on affine parameters.
    """
    if len(size) != dimension: raise Exception("size must have length {0}.".format(dimension))
    if len(spacing) != dimension: raise Exception("spacing must have length {0}.".format(dimension))

    # Set affine parameters
    affineTransform = sitk.AffineTransform(dimension)
    numParameters = len(affineTransform.GetParameters())
    if len(affine) != numParameters: raise Exception("affine must have length {0}.".format(numParameters))
    affineTransform = sitk.AffineTransform(dimension)
    affineTransform.SetParameters(affine)

    # Convert affine transform to field
    return  sitk.TransformToDisplacementFieldFilter().Execute(affineTransform, vectorType, size, zeroOrigin, spacing, identityDirection)

def airAffineToField(airTxT, size, spacing):
    ScanAirOutputFileHandle = open(airTxT,"r")
    ScanAirOutputFile = ScanAirOutputFileHandle.read()
    ParametersBeginIndex = ScanAirOutputFile.index("[")+1 
    ParametersEndIndex = ScanAirOutputFile.index("]") 
    P = ScanAirOutputFile[ParametersBeginIndex:ParametersEndIndex].split()  

    for dd in xrange(0,3):
        del P[-1]

    P1 = []    
    for dd in xrange(0,12):
        P1.append(float(P[dd]))

    b1 = P1[3]
    b2 = P1[7]
    b3 = P1[11]

    for dd in [11,7,3]:
        del P1[dd]

    P1.append(b1)
    P1.append(b2)
    P1.append(b3)

    airField = affineToField(P1, size, spacing)
    return  airField

def airApplyField(airTxT, field, size, spacing):
    airField = airAffineToField(airTxT, size, spacing)
    combineField = fieldApplyField(field, airField)
    return combineField

def imgApplyField(img, field, useNearest=False, size=[], spacing=[],defaultValue=0):
    """
    img \circ field
    """
    field = sitk.Cast(field, sitk.sitkVectorFloat64)

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]

    # Set transform field
    transform = sitk.DisplacementFieldTransform(img.GetDimension())
    transform.SetInterpolator(sitk.sitkLinear)
    transform.SetDisplacementField(field)

    # Set size
    if size == []:
        size = img.GetSize()
    else:
        if len(size) != img.GetDimension(): raise Exception("size must have length {0}.".format(img.GetDimension()))

    # Set Spacing
    if spacing == []:
        spacing = img.GetSpacing()
    else:
        if len(spacing) != img.GetDimension(): raise Exception("spacing must have length {0}.".format(img.GetDimension()))
    
    # Apply displacement transform
    return  sitk.Resample(img, size, transform, interpolator, [0]*img.GetDimension(), spacing, img.GetDirection() ,defaultValue)


def fieldApplyField(inField, field):
    """ outField = inField \circ field """
    inField = sitk.Cast(inField, sitk.sitkVectorFloat64)
    field = sitk.Cast(field, sitk.sitkVectorFloat64)
    
    size = list(inField.GetSize())
    spacing = list(inField.GetSpacing())

    # Create transform for input field
    inTransform = sitk.DisplacementFieldTransform(dimension)
    inTransform.SetDisplacementField(inField)
    inTransform.SetInterpolator(sitk.sitkLinear)

    # Create transform for field
    transform = sitk.DisplacementFieldTransform(dimension)
    transform.SetDisplacementField(field)
    transform.SetInterpolator(sitk.sitkLinear)
    
    # Combine thransforms
    outTransform = sitk.Transform()
    outTransform.AddTransform(transform)
    outTransform.AddTransform(inTransform)

    # Get output displacement field
    return sitk.TransformToDisplacementFieldFilter().Execute(outTransform, vectorType, size, zeroOrigin, spacing, identityDirection)

def imgHM(inImg, refImg, numMatchPoints=64, numBins=256):
    """
    Histogram matches input image to reference image and writes result to output image
    """
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    return  sitk.HistogramMatchingImageFilter().Execute(inImg, refImg, numBins, numMatchPoints, False)

def imgChecker(inImg, refImg, useHM=True, pattern=[4]*dimension):
    """
    Checkerboards input image with reference image
    """    
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    inSize = list(inImg.GetSize())
    refSize = list(refImg.GetSize())

    if(inSize != refSize):
        sourceSize = np.array([inSize, refSize]).min(0)
        tmpImg = sitk.Image(refSize,refImg.GetPixelID()) # Empty image with same size as reference image
        tmpImg.CopyInformation(refImg)
        inImg = sitk.PasteImageFilter().Execute(tmpImg, inImg, sourceSize, zeroIndex, zeroIndex)

    if useHM: inImg = imgHM(inImg, refImg)

    return sitk.CheckerBoardImageFilter().Execute(inImg, refImg,pattern)

def imgMetamorphosis(inImg, refImg, alpha=0.02, scale=1.0, iterations=1000, MASGD=False, useNearest=False, Metric=0, verbose=False, debug=False, outDirPath=""):
    """
    Performs lddmmMSGD registration between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)
        
    inPath = outDirPath + "in.img"
    imgWrite(inImg, inPath)
    refPath = outDirPath + "ref.img"
    imgWrite(refImg, refPath)
    outPath = outDirPath + "out.img"

    fieldPath = outDirPath + "field.vtk"
    
    binPath = lddmmMASGDDirPath + "lddmmMASGD" 
    steps = 2 
    command = binPath + " --in {0} --ref {1} --out {2} --alpha {3} --field {4} --iterations {5} --scale {6} --steps {7} --verbose {8}".format(inPath, refPath, outPath, alpha, fieldPath, iterations, scale, steps, np.int(verbose))
    
    if(Metric!=1 and Metric!=2):
	    command += " --cost 0  --epsilon 0.05 "
    if(Metric==1):
        command += " --cost 1  --epsilon 0.05 " 
    if(Metric==2):
        command += " --cost 2 --epsilon 0.05 --bins 32 "
    if(MASGD==True):
        command += " --MASGD 1 "

    
    if debug: print(command)
    logText = run_shell_command(command, verbose=verbose)
    logPath = outDirPath+"log.txt"
    txtWrite(logText, logPath)

    field = imgRead(fieldPath)
    if useTempDir: shutil.rmtree(outDirPath)

    return field

def imgMetamorphosisComposite(inImg, refImg, alphaList=0.02, scaleList=1.0, Metric=0, iterations=1000, MASGD=False, useNearest=False, verbose=True, debug=False, outDirPath=""):
    """
    Performs lddmmMSGD between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        if(Metric==1):
            outDirPath = outDirPath+"lddmmcc"
        elif(Metric==2):
            outDirPath = outDirPath+"lddmmmi"
        else:
            outDirPath = outDirPath+"lddmmssd"
        outDirPath = dirMake(outDirPath)

    if isNumber(alphaList): alphaList = [float(alphaList)]
    if isNumber(scaleList): scaleList = [float(scaleList)]
    if isNumber(iterations): iterations = [float(iterations)]
   
    numSteps = max(len(alphaList), len(scaleList), len(iterations))

    if len(alphaList) != numSteps:
        if len(alphaList) != 1:
            raise Exception("Legth of alphaList must be 1 or same length as betaList")
        else:
            alphaList *= numSteps
        
    if len(scaleList) != numSteps:
        if len(scaleList) != 1:
            raise Exception("Legth of scaleList must be 1 or same length as alphaList")
        else:
            scaleList *= numSteps

    if len(iterations) != numSteps:
	      if len(iterations) != 1:
	          raise Exception("Length of scaleList must be 1 or same length as alphalist")
	      else:
	          iterations *= numSteps
    
    origInImg = inImg
    for step in range(numSteps):
        alpha = alphaList[step]
        scale = scaleList[step]
        iteration = iterations[step]

        stepDirPath = outDirPath + "step" + str(step) + "/"
        #if(verbose): 
        print("\nStep {0}: alpha={1}, scale={2}, iteration={3}\n".format(step, alpha, scale, iteration))
        field = imgMetamorphosis(inImg, 
                                 refImg, 
                                 alpha, 
                                 scale, 
                                 iteration, 
                                 MASGD,
                                 useNearest, 
                                 Metric,
                                 verbose,
                                 debug,
                                 outDirPath=stepDirPath)

        if step == 0:
            compositeField = field
        else:
            compositeField = fieldApplyField(field, compositeField)

            if outDirPath != "":
                fieldPath = stepDirPath+"field.vtk"
                imgWrite(compositeField, fieldPath)

        inImg = imgApplyField(origInImg, compositeField, size=refImg.GetSize())

    # Write final results
    if outDirPath != "":
        imgWrite(compositeField, outDirPath+"field.vtk")
        imgWrite(inImg, outDirPath+"out.img")
        imgWrite(imgChecker(inImg,refImg), outDirPath+"checker.img")
    
    if useTempDir: shutil.rmtree(outDirPath)
    return compositeField

def imgAffine(inImgDir, refImgDir, outDirPath=""):
    """
    Performs affine registration between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)

    inImgName = os.path.basename(inImgDir)[:-4]
    refImgName = os.path.basename(refImgDir)[:-4]
    
    affinedImgName = inImgName+'_to_'+refImgName
    affinedDir = outDirPath+affinedImgName+'/'

    #  the path of air registration executable file
    airbin = airBinPath
    airscript = airBinPath+'AIR_registration1_1024.pl'

    #print(airscript+' '+airbin+' '+inImgDir+' '+refImgDir+' '+affinedDir+' 1 '+affinedImgName+'_affined.img')
    subprocess.call([airscript, airbin, inImgDir, refImgDir, affinedDir, '1', affinedImgName+'_affined.img'])
    airtxt = affinedDir+affinedImgName+'_affined_air.txt'
    affinedImageDir = affinedDir+affinedImgName+'_affined.img'

    return affinedImageDir, airtxt


def imgHistMatchOrgName(inImgDir, refImgDir, outDirPath=""):
    """
    Performs Histogram Matching between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)
    
    
    inImgName = os.path.basename(inImgDir)[:-4]
    refImgName = os.path.basename(refImgDir)[:-4]
    HistMatchedName = inImgName+'_to_'+refImgName
    #outDirPath = outDirPath+'/'+HistMatchedName+'/'
    dirMake(outDirPath)

    inHistMatched = outDirPath+inImgName+'.img'
    refHistMatched = outDirPath+refImgName+'.img'
    inHist1 = outDirPath+HistMatchedName+'_hist1'
    inHist2 = outDirPath+HistMatchedName+'_hist2'
    refHist1 = outDirPath+refImgName+'_hist1'
    refHist2 = outDirPath+refImgName+'_hist2'

    #  the path of histogram matching executable file
    histMatchBin = histMatchBinPath+'IMG_histmatch4'
    subprocess.call([histMatchBin, inImgDir, refImgDir, inHistMatched, refHistMatched, '1024', '3', '0', '1', inHist1, refHist1, inHist2, refHist2])
    
    delFile(inHist1)
    delFile(inHist2)
    delFile(refHist1)
    delFile(refHist2)

    return inHistMatched, refHistMatched

def imgHistMatch(inImgDir, refImgDir, outDirPath=""):
    """
    Performs Histogram Matching between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)
    
    if '.nii.gz' in inImgDir:
        inImgName = os.path.basename(inImgDir)[:-7]
    else:
        inImgName = os.path.basename(inImgDir)[:-4]
        
    if '.nii.gz' in refImgDir:
        refImgName = os.path.basename(refImgDir)[:-7]
    else:
        refImgName = os.path.basename(refImgDir)[:-4]
    
    HistMatchedName = inImgName+'_to_'+refImgName
    outDirPath = outDirPath+'/'+HistMatchedName+'/'
    dirMake(outDirPath)

    inHistMatched = outDirPath+HistMatchedName+'_histMatched.img'
    refHistMatched = outDirPath+refImgName+'_histMatched.img'
    inHist1 = outDirPath+HistMatchedName+'_hist1'
    inHist2 = outDirPath+HistMatchedName+'_hist2'
    refHist1 = outDirPath+refImgName+'_hist1'
    refHist2 = outDirPath+refImgName+'_hist2'

    print(inImgDir)
    print(refImgDir)
    #  the path of histogram matching executable file
    histMatchBin = histMatchBinPath+'IMG_histmatch4'
    subprocess.call([histMatchBin, inImgDir, refImgDir, inHistMatched, refHistMatched, '1024', '3', '0', '1', inHist1, refHist1, inHist2, refHist2])
    
    delFile(inHist1)
    delFile(inHist2)
    delFile(refHist1)
    delFile(refHist2)

    return inHistMatched, refHistMatched

def imgRegistration(inImgDir, refImgDir, alphaList=0.02, scaleList=1.0, Metric=0, iterations=1000, MASGD=False, useNearest=False, verbose=True, debug=False, reOrient=False, inImgOrient=None, refImgOrient=None, imgBias=False, outIniteDir="", outAffineDir="", outHistDir="", outlddmmDir="", outFieldDir=""):

    print ("\n===================================================")
    print ("*************** Do the preprocessing **************")
    print ("===================================================\n\n")
    if '.nii.gz' in inImgDir:
        inImgName = os.path.basename(inImgDir)[:-7]
    else:
        inImgName = os.path.basename(inImgDir)[:-4]
        
    if '.nii.gz' in refImgDir:
        refImgName = os.path.basename(refImgDir)[:-7]
    else:
        refImgName = os.path.basename(refImgDir)[:-4]
        
    inImg = imgRead(inImgDir)
    refImg = imgRead(refImgDir)
    #print(inImg.GetSpacing(), refImg.GetSpacing())

    if reOrient:
        if inImgOrient==None or refImgOrient==None:
            raise Exception(
                "both in_orient and ref_orient must be supported ")
        else:
            inImg = Image_orientation(inImg, inImg_orientation=inImgOrient, refImg_orientation=refImgOrient)

    if imgBias:
        inImg, refImg = Image_bias(inImg, refImg)

    if (inImg.GetSpacing() != (1.0,1.0,1.0) or refImg.GetSpacing() != (1.0,1.0,1.0)):
        print("resampling operation to make spacing of image to be [1.0,1.0,1.0]")
        fixedSpacing = [1.0, 1.0, 1.0]
        inImg = imgResample(inImg, fixedSpacing)
        refImg = imgResample(refImg, fixedSpacing)

    if (inImg.GetSize() != refImg.GetSize()):
        print("resampling operation to make size of moving image to be the same as the reference image")
        inImg = imgResample(inImg, spacing=refImg.GetSpacing(), size=refImg.GetSize())

    imgWrite(inImg, outIniteDir+inImgName+'.img')
    imgWrite(refImg, outIniteDir+refImgName+'.img')

    print ("\n===================================================")
    print ("*********** Do the Affine Registration ************")
    print ("===================================================\n\n")
    inImgDir = outIniteDir + inImgName + '.img'
    refImgDir = outIniteDir + refImgName + '.img'

    affinedImageDir, airtxt = imgAffine(inImgDir, refImgDir, outAffineDir)
    tempName = inImgName+'_to_'+refImgName


    print ("\n===================================================")
    print ("*********** Do the Histogram Matching *************")
    print ("===================================================\n\n") 


    inHistMatchedDir, refHistMatchedDir = imgHistMatch(affinedImageDir, refImgDir, outHistDir)

    print ("\n===================================================")
    print ("*********** Do the lddmmMASGD Registration ********")
    print ("===================================================\n") 
    print("1> Template image: ",inHistMatchedDir)
    print("2> Target img: ",refHistMatchedDir)
    if (Metric==1):
        print("3> Metric: Cross Correlation\n")
    elif (Metric==2):
        print("3> Metric: Mutual Information\n")
    else:
        print("3> Metric: Sum of Squred Difference\n")
    inImg = imgRead(inHistMatchedDir)
    refImg = imgRead(refHistMatchedDir)

    outlddmmDir = outlddmmDir+tempName+'/'
    field = imgMetamorphosisComposite(inImg, refImg, alphaList, scaleList, Metric, iterations, MASGD, useNearest, verbose, debug, outDirPath=outlddmmDir)
    
    combineField =airApplyField(airtxt, field, refImg.GetSize(), refImg.GetSpacing())

    if outFieldDir == "":
        outFieldDir = tempfile.mkdtemp() + "/"
    else:
        outFieldDir = dirMake(outFieldDir)

    airField = airAffineToField(airtxt, refImg.GetSize(), refImg.GetSpacing())
    imgWrite(airField, outFieldDir+tempName+'_affined.vtk')
    imgWrite(combineField, outFieldDir+tempName+'_combine.vtk')

    return combineField

def imgApplyFieldSave(inImgDir, refImgDir, fieldDir, reOrient=False, imgBias=False, inImgOrient=None, refImgOrient=None, outDirPath=""):
    if outDirPath == "":
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)
    
    if '.nii.gz' in inImgDir:
        inImgName = os.path.basename(inImgDir)[:-7]
    else:
        inImgName = os.path.basename(inImgDir)[:-4]
        
    if '.nii.gz' in refImgDir:
        refImgName = os.path.basename(refImgDir)[:-7]
    else:
        refImgName = os.path.basename(refImgDir)[:-4]
        
    tempImgName = inImgName+'_to_'+refImgName
    saveImage = outDirPath + '/' + tempImgName + '.nii.gz'
    field = fieldDir + tempImgName+'_combine.vtk'

    print ("\n===================================================")
    print ("*********** apply field to image and saving *******")
    print ("===================================================\n") 
    print("1> image: ",inImgDir)
    print("2> save file: ",saveImage)
    print("3> field:",field)
    print("\n\n")


    field = imgRead(field)
    inImage = imgRead(inImgDir)
    refImage = imgRead(refImgDir)

    if reOrient:
        if inImgOrient == None or refImgOrient == None:
            raise Exception(
                "both in_orient and ref_orient must be supported ")
        else:
            inImage = Image_orientation(inImage, inImg_orientation=inImgOrient, refImg_orientation=refImgOrient)

    if imgBias:
        inImage, refImage = Image_bias(inImage, refImage)

    if (inImage.GetSpacing() != (1.0, 1.0, 1.0) or refImage.GetSpacing() != (1.0, 1.0, 1.0)):
        print("resampling operation to make spacing of image to be [1.0,1.0,1.0]")
        fixedSpacing = [1.0, 1.0, 1.0]
        inImage = imgResample(inImage, fixedSpacing)
        refImage = imgResample(refImage, fixedSpacing)

    if (inImage.GetSize() != refImage.GetSize()):
        print("resampling operation to make size of moving image to be the same as the reference image")
        inImage = imgResample(inImage, spacing=refImage.GetSpacing(), size=refImage.GetSize())

    if (inImage.GetSize() != refImage.GetSize()):
        print("resampling operation")
        inImage = imgResample(inImage, spacing=refImage.GetSpacing(), size=refImage.GetSize())
    
    alginedImage = imgApplyField(inImage,field,useNearest=False,size=refImage.GetSize(),spacing=refImage.GetSpacing())
    refImage = sitk.ReadImage(refImgDir)
    if (inImage.GetSpacing() != refImage.GetSpacing()):
        alginedImage = imgResample(alginedImage, spacing=refImage.GetSpacing(), size=refImage.GetSize())    
    alginedImage.SetDirection(refImage.GetDirection())
    imgWrite(alginedImage, saveImage)
    

def labApplyFieldSave(inLabelsDir, refLabelsDir, inImgDir, refImgDir, fieldDir, reOrient=False, inImgOrient=None, refImgOrient=None, outDirPath=""):
    if outDirPath == "":
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)

    if '.nii.gz' in inLabelsDir:
        inImgName = os.path.basename(inLabelsDir)[:-7]
    else:
        inImgName = os.path.basename(inLabelsDir)[:-4]
        
    if '.nii.gz' in refLabelsDir:
        refImgName = os.path.basename(refLabelsDir)[:-7]
    else:
        refImgName = os.path.basename(refLabelsDir)[:-4]
        
    tempImgName = inImgName + '_to_' + refImgName
    saveLabels = outDirPath + '/' + tempImgName + '-label.nii.gz'
    fieldDir = fieldDir + tempImgName + '_combine.vtk'

    print ("\n===================================================")
    print ("*********** apply field to labels and save ********")
    print ("===================================================\n") 
    print("1> image: ",inLabelsDir)
    print("2> save file: ",saveLabels)
    print("3> fiel:", fieldDir)
    print("\n\n")


    field = imgRead(fieldDir)
    inLabels = imgRead(inLabelsDir)
    refLabels = imgRead(refLabelsDir)

    if (inLabels.GetSpacing() != (1.0, 1.0, 1.0) or refLabels.GetSpacing() != (1.0, 1.0, 1.0)):
        print("resampling operation to make spacing of image to be [1.0,1.0,1.0]")
        fixedSpacing = [1.0, 1.0, 1.0]
        inLabels = imgResample(inLabels, fixedSpacing, useNearest=True)
        refLabels = imgResample(refLabels, fixedSpacing, useNearest=True)

    if (inLabels.GetSize() != refLabels.GetSize()):
        print("resampling operation to make size of moving image to be the same as the reference image")
        inLabels = imgResample(inLabels, spacing=refLabels.GetSpacing(), size=refLabels.GetSize(), useNearest=True)
        imgWrite(inLabels, saveLabels)
        inLabels = imgRead(saveLabels)

    alignedLabels = imgApplyField(inLabels, field, useNearest=True, size=refLabels.GetSize(),
                                  spacing=refLabels.GetSpacing())
    refLabels = sitk.ReadImage(refLabelsDir)
    if (inLabels.GetSpacing() != refLabels.GetSpacing() or inLabels.GetSize() != refLabels.GetSize()):
        alignedLabels = imgResample(alignedLabels, useNearest=True, spacing=refLabels.GetSpacing(),
                                    size=refLabels.GetSize())
    alignedLabels.SetDirection(refLabels.GetDirection())
    alignedLabels.SetOrigin(refLabels.GetOrigin())
    imgWrite(alignedLabels, saveLabels)

def imgSlices(img, flip=[0,0,0], numSlices=1):
   size = img.GetSize()
   sliceImgList = []
   for i in range(img.GetDimension()):
       start = size[2-i]/(numSlices+1)
       sliceList = np.linspace(start, size[2-i]-start, numSlices)
       sliceSize = list(size)
       sliceSize[2-i] = 0
       
       for (j, slice) in enumerate(sliceList):
           sliceIndex = [0]*img.GetDimension()
           sliceIndex[2-i] = int(slice)
           sliceImg = sitk.Extract(img, sliceSize, sliceIndex)
           
           if flip[i]:
               sliceImgDirection = sliceImg.GetDirection()
               sliceImg = sitk.PermuteAxesImageFilter().Execute(sliceImg, range(sliceImg.GetDimension()-1,-1,-1))
               sliceImg.SetDirection(sliceImgDirection)
           sliceImgList.append(sliceImg)

   return sliceImgList


def imgShow(img, vmin=None, vmax=None, cmap=None, alpha=None,
            newFig=True, flip=None, numSlices=3, useNearest=False):
    """
    Displays an image.  Only 2D images are supported for now
    """
    if flip == None: flip = [0, 0, 0]
    if newFig: plt.figure()

    if (vmin is None) or (vmax is None):
        stats = sitk.StatisticsImageFilter()
        stats.Execute(img)
        if vmin is None:
            vmin = stats.GetMinimum()
        if vmax is None:
            vmax = stats.GetMaximum()

    if cmap is None:
        cmap = plt.cm.gray
    if alpha is None:
        alpha = 1.0

    interpolation = ['bilinear', 'none'][useNearest]

    if img.GetDimension() == 2:
        plt.axis('off')
        ax = plt.imshow(sitk.GetArrayFromImage(img), cmap=cmap, vmin=vmin,
                        vmax=vmax, alpha=alpha, interpolation=interpolation)

    elif img.GetDimension() == 3:
        size = img.GetSize()
        for i in range(img.GetDimension()):
            start = size[2 - i] / (numSlices + 1)
            sliceList = np.linspace(start, size[2 - i] - start, numSlices)
            sliceSize = list(size)
            sliceSize[2 - i] = 0

            for (j, slice_) in enumerate(sliceList):
                sliceIndex = [0] * img.GetDimension()
                sliceIndex[2 - i] = int(slice_)
                sliceImg = sitk.Extract(img, sliceSize, sliceIndex)
                sliceArray = sitk.GetArrayFromImage(sliceImg)
                if flip[i]:
                    sliceArray = np.transpose(sliceArray)

                plt.subplot(numSlices, img.GetDimension(),
                            i + img.GetDimension() * j + 1)
                ax = plt.imshow(sliceArray, cmap=cmap, vmin=vmin,
                                vmax=vmax, alpha=alpha, interpolation=interpolation)
                plt.axis('off')
    else:
        raise Exception("Image dimension must be 2 or 3.")

    if newFig:
        plt.show()
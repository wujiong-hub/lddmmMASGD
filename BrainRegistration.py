#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import os, math, sys, subprocess, tempfile, shutil
from itertools import product
import matplotlib.pyplot as plt

lddmmMASGDDirPath = os.path.dirname(os.path.realpath(__file__))+"/"
dimension = 3
vectorComponentType = sitk.sitkFloat32
vectorType = sitk.sitkVectorFloat32
affine = sitk.AffineTransform(dimension)
identityAffine = list(affine.GetParameters())
identityDirection = list(affine.GetMatrix())
zeroOrigin = [0]*dimension
zeroIndex = [0]*dimension

ndToSitkDataTypes = {'uint8': sitk.sitkUInt8,
                     'uint16': sitk.sitkUInt16,
                     'uint32': sitk.sitkUInt32,
                     'float32': sitk.sitkFloat32}


sitkToNpDataTypes = {sitk.sitkUInt8: np.uint8,
                     sitk.sitkUInt16: np.uint16,
                     sitk.sitkUInt32: np.uint32,
                     sitk.sitkInt8: np.int8,
                     sitk.sitkInt16: np.int16,
                     sitk.sitkInt32: np.int32,
                     sitk.sitkFloat32: np.float32,
                     sitk.sitkFloat64: np.float64,
                     }


ndregDirPath = './'
ndregTranslation = 0
ndregRigid = 1
ndregAffine = 2 

def isIterable(obj):
    """
    Returns True if obj is a list, tuple or any other iterable object
    """
    return hasattr([],'__iter__')

def isNumber(variable):
    try:
        float(variable)
    except TypeError:
        return False
    return True

def run(command, checkReturnValue=True, verbose=False):
    print (command)
    process = subprocess.Popen(command, shell=True)
    process.communicate()[0]
    returnValue = process.returncode
    if checkReturnValue and (returnValue != 0): raise Exception(outText)
    return returnValue

def run_shell_command(command, checkReturnValue=True, verbose=False):
    """
    Runs a shell command and returns the output.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1)
    outText = ""

    for line in iter(process.stdout.readline, ''):
        if verbose:
            sys.stdout.write(line)
        outText += line

    process.communicate()[0]
    """
    returnValue = process.returncode
    if checkReturnValue and (returnValue != 0):
       raise Exception(outText)
    """
    return outText

def txtWrite(text, path, mode="w"):
    """
    Conveinence function to write text to a file at specified path
    """
    dirMake(os.path.dirname(path))
    textFile = open(path, mode)
    print(text, file=textFile)
    textFile.close()

def txtRead(path):
    """
    Conveinence function to read text from file at specified path
    """
    textFile = open(path,"r")
    text = textFile.read()
    textFile.close()
    return text

def txtReadList(path):
    return map(float,txtRead(path).split())

def txtWriteList(parameterList, path):
    txtWrite(" ".join(map(str,parameterList)), path)

def dirMake(dirPath):
    if dirPath != "":
        if not os.path.exists(dirPath): os.makedirs(dirPath)
        return os.path.normpath(dirPath) + "/"
    else:
        return dirPath


def imgHM(inImg, refImg, numMatchPoints=64, numBins=256):
    """
    Histogram matches input image to reference image and writes result to output image
    """
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    return  sitk.HistogramMatchingImageFilter().Execute(inImg, refImg, numBins, numMatchPoints, False)

def imgRead(path):
    """
    Alias for sitk.ReadImage
    """

    inImg = sitk.ReadImage(path)
    inImg = imgCollaspeDimension(inImg) ###
    if(inImg.GetDimension() == 2): inImg = sitk.JoinSeriesImageFilter().Execute(inImg)
        
    inDimension = inImg.GetDimension()
    inImg.SetDirection(sitk.AffineTransform(inDimension).GetMatrix())
    inImg.SetOrigin([0]*inDimension)

    return inImg

def imgCopy(img):
    """
    Returns a copy of the input image
    """
    return sitk.Image(img)

def imgWrite(img, path):
    """
   # Write sitk image to path.
    """
    dirMake(os.path.dirname(path))
    sitk.WriteImage(img, path)


def imgResample(img, spacing, size=[], useNearest=False):
    """
    Resamples image to given spacing and size.
    """
    if len(spacing) != img.GetDimension(): raise Exception("len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size == []:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [int(math.ceil(inSize[i]*(inSpacing[i]/spacing[i]))) for i in range(img.GetDimension())]
    else:
        if len(size) != img.GetDimension(): raise Exception("len(size) != " + str(img.GetDimension()))
    
    # Resample input image
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    identityTransform = sitk.Transform()
    
    return sitk.Resample(img, size, identityTransform, interpolator, zeroOrigin, spacing)

def imgPad(img, padding=0, useNearest=False):
     """
     Pads image by given ammount of padding in units spacing.
     For example if the input image has a voxel spacing of 0.5 and the padding=2.0 then the image will be padded by 4 voxels.
     If the padding < 0 then the filter crops the image
     """
     if isNumber(padding):
          padding = [padding]*img.GetDimension()
     elif len(padding) != img.GetDimension():
          raise Exception("padding must have length {0}.".format(img.GetDimension()))
     
     interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
     translationTransform = sitk.TranslationTransform(img.GetDimension(), -np.array(padding))
     spacing = img.GetSpacing()
     size = list(img.GetSize())
     for i in range(img.GetDimension()):
          if padding[i] > 0:
               paddingVoxel = int(math.ceil(2*padding[i] / spacing[i]))
          else:
               paddingVoxel = int(math.floor(2*padding[i] / spacing[i]))
          size[i]+=paddingVoxel

     origin = [0]*img.GetDimension()
     return sitk.Resample(img, size, translationTransform, interpolator, origin, spacing)

def imgLargestMaskObject(maskImg):
    ccFilter = sitk.ConnectedComponentImageFilter()
    labelImg = ccFilter.Execute(maskImg)
    numberOfLabels = ccFilter.GetObjectCount()
    labelArray = sitk.GetArrayFromImage(labelImg)
    labelSizes = np.bincount(labelArray.flatten())
    largestLabel = np.argmax(labelSizes[1:])+1
    outImg = sitk.GetImageFromArray((labelArray==largestLabel).astype(np.int16))
    outImg.CopyInformation(maskImg) # output image should have same metadata as input mask image
    return outImg

def createTmpRegistration(inMask=None, refMask=None, samplingFraction=1.0, dimension=dimension):
    identityTransform = sitk.Transform(dimension, sitk.sitkIdentity)
    tmpRegistration = sitk.ImageRegistrationMethod()
    tmpRegistration.SetInterpolator(sitk.sitkNearestNeighbor)
    tmpRegistration.SetInitialTransform(identityTransform)
    tmpRegistration.SetOptimizerAsGradientDescent(learningRate=1e-14, numberOfIterations=1)
    if samplingFraction != 1.0:
        tmpRegistration.SetMetricSamplingPercentage(samplingFraction)
        tmpRegistration.SetMetricSamplingPercentage(tmpRegistration.RANDOM)

    if(inMask): tmpRegistration.SetMetricMovingMask(inMask)
    if(refMask): tmpRregistration.SetMetricFixedMask(refMask)

    return tmpRegistration

def imgCollaspeDimension(inImg):
    inSize = inImg.GetSize()

    if inImg.GetDimension() == dimension and inSize[dimension-1] == 1:
        outSize = list(inSize)
        outSize[dimension-1] = 0
        outIndex = [0]*dimension
        inImg = sitk.Extract(inImg, outSize, outIndex, 1)
        
    return inImg


def imgNorm(img):
    """
    Returns the L2-Norm of an image
    """
    if img.GetNumberOfComponentsPerPixel() > 1: img = sitk.VectorMagnitude(img)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)    
    return stats.GetSum()


def imgMI(inImg, refImg, inMask=None, refMask=None, numBins=128, samplingFraction=1.0):
    """
    Compute mattes mutual information between input and reference images
    """
        
    
    # In SimpleITK the metric can't be accessed directly.
    # Therefore we create a do-nothing registration method which uses an identity transform to get the metric value
    inImg = imgCollaspeDimension(inImg)
    refImg = imgCollaspeDimension(refImg)    
    if inMask: imgCollaspeDimension(inMask)
    if refMask: imgCollaspeDimension(refMask)

    tmpRegistration = createTmpRegistration(inMask, refMask, dimension=inImg.GetDimension(), samplingFraction=samplingFraction)
    tmpRegistration.SetMetricAsMattesMutualInformation(numBins)
    tmpRegistration.Execute( sitk.Cast(refImg,sitk.sitkFloat32),sitk.Cast(inImg, sitk.sitkFloat32) )

    return -tmpRegistration.GetMetricValue()

def imgMSE(inImg, refImg, inMask=None, refMask=None, samplingFraction=1.0):
    """
    Compute mean square error between input and reference images
    """
    inImg = imgCollaspeDimension(inImg)
    refImg = imgCollaspeDimension(refImg)    
    if inMask: imgCollaspeDimension(inMask)
    if refMask: imgCollaspeDimension(refMask)
    tmpRegistration = createTmpRegistration(inMask, refMask, dimension=refImg.GetDimension(), samplingFraction=1.0)
    tmpRegistration.SetMetricAsMeanSquares()
    tmpRegistration.Execute( sitk.Cast(refImg,sitk.sitkFloat32),sitk.Cast(inImg, sitk.sitkFloat32) )

    return tmpRegistration.GetMetricValue()


def imgMakeRGBA(imgList, dtype=sitk.sitkUInt8):
    if len(imgList) < 3 or len(imgList) > 4: raise Exception("imgList must contain 3 ([r,g,b]) or 4 ([r,g,b,a]) channels.")
    
    inDatatype = sitkToNpDataTypes[imgList[0].GetPixelID()]
    outDatatype = sitkToNpDataTypes[dtype]
    inMin = np.iinfo(inDatatype).min
    inMax = np.iinfo(inDatatype).max
    outMin = np.iinfo(outDatatype).min
    outMax = np.iinfo(outDatatype).max

    castImgList = []
    for img in imgList:
        castImg = sitk.Cast(sitk.IntensityWindowingImageFilter().Execute(img, inMin, inMax, outMin, outMax), dtype)
        castImgList.append(castImg)

    if len(imgList) == 3:        
        channelSize = list(imgList[0].GetSize())
        alphaArray = outMax*np.ones(channelSize[::-1], dtype=outDatatype)
        alphaChannel = sitk.GetImageFromArray(alphaArray)
        alphaChannel.CopyInformation(imgList[0])
        castImgList.append(alphaChannel)

    return sitk.ComposeImageFilter().Execute(castImgList)

def imgThreshold(img, threshold=0):
    """
    Thresholds image at inPath at given threshold and writes result to outPath.
    """
    return sitk.BinaryThreshold(img, 0, threshold, 0, 1)

def imgMakeMask(inImg, threshold=None, forgroundValue=1):
    
    if threshold is None:
        # Initialize binary mask using otsu threshold
        inMask = sitk.BinaryThreshold(inImg, 0, 0, 0, forgroundValue) # Mask of non-zero voxels
        otsuThresholder = sitk.OtsuThresholdImageFilter()
        otsuThresholder.SetInsideValue(0)
        otsuThresholder.SetOutsideValue(forgroundValue)
        otsuThresholder.SetMaskValue(forgroundValue)
        tmpMask = otsuThresholder.Execute(inImg, inMask)
    else:
        # initialzie binary mask using given threshold
        tmpMask = sitk.BinaryThreshold(inImg, 0, threshold, 0, forgroundValue)

    # Assuming input image is has isotropic resolution...
    # ... compute size of morphological kernels in voxels.
    spacing = min(list(inImg.GetSpacing()))
    openingRadiusMM = 0.05  # In mm
    closingRadiusMM = 0.2   # In mm
    openingRadius = max(1, int(round(openingRadiusMM / spacing))) # In voxels
    closingRadius = max(1, int(round(closingRadiusMM / spacing))) # In voxels

    # Morphological open mask remove small background objects
    opener = sitk.GrayscaleMorphologicalOpeningImageFilter()
    opener.SetKernelType(sitk.sitkBall)
    opener.SetKernelRadius(openingRadius)
    tmpMask = opener.Execute(tmpMask)

    # Morphologically close mask to fill in any holes
    closer = sitk.GrayscaleMorphologicalClosingImageFilter()
    closer.SetKernelType(sitk.sitkBall)
    closer.SetKernelRadius(closingRadius)
    outMask = closer.Execute(tmpMask)

    return imgLargestMaskObject(outMask)

def imgMask(img, mask):
    mask = imgResample(mask, img.GetSpacing(), img.GetSize(), useNearest=True)
    mask = sitk.Cast(mask, img.GetPixelID())
    return  sitk.MaskImageFilter().Execute(img, mask)

def sizeOut(inImg, transform, outSpacing):
    outCornerPointList = []
    inSize = inImg.GetSize()
    for corner in product((0,1), repeat=inImg.GetDimension()):
        inCornerIndex = np.array(corner)*np.array(inSize)
        inCornerPoint = inImg.TransformIndexToPhysicalPoint(inCornerIndex)
        outCornerPoint = transform.GetInverse().TransformPoint(inCornerPoint)
        outCornerPointList += [list(outCornerPoint)]

    size = np.ceil(np.array(outCornerPointList).max(0) / outSpacing).astype(int)
    return size

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
    
def imgApplyAffine(inImg, affine, useNearest=False, size=[], spacing=[]):
    inDimension = inImg.GetDimension()

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]

    # Set affine parameters
    affineTransform = sitk.AffineTransform(inDimension)
    numParameters = len(affineTransform.GetParameters())
    if (len(affine) != numParameters): raise Exception("affine must have length {0}.".format(numParameters))
    affineTransform = sitk.AffineTransform(inDimension)
    affineTransform.SetParameters(affine)

    # Set Spacing
    if spacing == []:
        spacing = inImg.GetSpacing()
    else:
        if len(spacing) != inDimension: raise Exception("spacing must have length {0}.".format(inDimension))

    # Set size
    if size == []:
        # Compute size to contain entire output image
        size = sizeOut(inImg, affineTransform, spacing)
    else:
       if len(size) != inDimension: raise Exception("size must have length {0}.".format(inDimension))
    
    # Apply affine transform
    outImg = sitk.Resample(inImg, size, affineTransform, interpolator, zeroOrigin, spacing)

    return outImg


def affineInverse(affine):
    # x0 = A0*x1 + b0
    # x1 = (A0.I)*x0 + (-A0.I*b0) = A1*x0 + b1
    A0 = np.mat(affine[0:9]).reshape(3,3)
    b0 = np.mat(affine[9:12]).reshape(3,1)

    A1 = A0.I
    b1 = -A1*b0
    return A1.flatten().tolist()[0] + b1.flatten().tolist()[0]

def affineApplyAffine(inAffine, affine):
    """ A_{outAffine} = A_{inAffine} \circ A_{affine} """
    if (not(isIterable(inAffine))) or (len(inAffine) != 12): raise Exception("inAffine must be a list of length 12.")
    if (not(isIterable(affine))) or (len(affine) != 12): raise Exception("affine must be a list of length 12.")
    A0 = np.array(affine[0:9]).reshape(3,3)
    b0 = np.array(affine[9:12]).reshape(3,1)
    A1 = np.array(inAffine[0:9]).reshape(3,3)
    b1 = np.array(inAffine[9:12]).reshape(3,1)

    # x0 = A0*x1 + b0
    # x1 = A1*x2 + b1
    # x0 = A0*(A1*x2 + b1) + b0 = (A0*A1)*x2 + (A0*b1 + b0)
    A = np.dot(A0,A1)
    b = np.dot(A0,b1) + b0

    outAffine = A.flatten().tolist() + b.flatten().tolist()
    return outAffine

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

def imgReorient(inImg, inOrient, outOrient):
    """
    Reorients image from input orientation inOrient to output orientation outOrient.
    inOrient and outOrient must be orientation strings specifying the orientation of the image.
    For example an orientation string of "las" means that the ...
        x-axis increases from \"l\"eft to right
        y-axis increases from \"a\"nterior to posterior
        z-axis increases from \"s\"uperior to inferior
    Thus using inOrient = "las" and outOrient = "rpi" reorients the input image from left-anterior-superior to right-posterior-inferior orientation.
    """

    if (len(inOrient) != dimension) or not isinstance(inOrient, basestring): raise Exception("inOrient must be a string of length {0}.".format(dimension))
    if (len(outOrient) != dimension) or not isinstance(outOrient, basestring): raise Exception("outOrient must be a string of length {0}.".format(dimension))
    inOrient = str(inOrient).lower()
    outOrient = str(outOrient).lower()
    
    inDirection = ""
    outDirection = ""
    orientToDirection = {"r":"r", "l":"r", "s":"s", "i":"s", "a":"a", "p":"a"}
    for i in range(dimension):
        try:
            inDirection += orientToDirection[inOrient[i]]
        except:
            raise Exception("inOrient \'{0}\' is invalid.".format(inOrient))

        try:
            outDirection += orientToDirection[outOrient[i]]
        except:
            raise Exception("outOrient \'{0}\' is invalid.".format(outOrient))
        
    if len(set(inDirection)) != dimension: raise Exception("inOrient \'{0}\' is invalid.".format(inOrient))
    if len(set(outDirection)) != dimension: raise Exception("outOrient \'{0}\' is invalid.".format(outOrient))

    order = []
    flip = []
    for i in range(dimension):
        j = outDirection.find(inDirection[i])
        order += [j]
        flip += [inOrient[i] != outOrient[j]]

    outImg = sitk.FlipImageFilter().Execute(inImg, flip, False)
    outImg = sitk.PermuteAxesImageFilter().Execute(outImg, order)
    outImg.SetDirection(identityDirection)
    outImg.SetOrigin(zeroOrigin)

    return outImg

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

def imgAffine(inImg, refImg, method=ndregAffine, scale=1.0, useNearest=False, useMI=False, iterations=1000, inMask=None, refMask=None, verbose=False):
    """
    Perform Affine Registration between input image and reference image
    """
    inDimension = inImg.GetDimension()

    # Rescale images
    refSpacing = refImg.GetSpacing()
    spacing = [x / scale for x in refSpacing]
    inImg = imgResample(inImg, spacing, useNearest=useNearest)
    refImg = imgResample(refImg, spacing, useNearest=useNearest)
    if(inMask): imgResample(inMask, spacing, useNearest=False)
    if(refMask): imgResample(refMask, spacing, useNearest=False)

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    
    # Set transform
    try:
        rigidTransformList = [sitk.Similarity2DTransform(), sitk.Similarity3DTransform()]
        transform = [sitk.TranslationTransform(inDimension), rigidTransformList[inDimension-2], sitk.AffineTransform(inDimension)][method]
    except:
        raise Exception("method is invalid")

    # Do registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetInterpolator(interpolator)
    registration.SetInitialTransform(transform)

    if(inMask): registration.SetMetricMovingMask(inMask)
    if(refMask): registration.SetMetricFixedMask(refMask)
    
    if useMI:
        numHistogramBins = 64
        registration.SetMetricAsMattesMutualInformation(numHistogramBins)
 
    else:
        registration.SetMetricAsMeanSquares()

    learningRate=0.1


    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=learningRate, numberOfIterations=iterations, estimateLearningRate=registration.EachIteration,minStep=0.001)
    if(verbose): registration.AddCommand(sitk.sitkIterationEvent, lambda: print("{0}.\t {1}".format(registration.GetOptimizerIteration(),registration.GetMetricValue())))

    ### if method == ndregRigid: registration.SetOptimizerScales([1,1,1,1,1,1,0.1])
                    
    registration.Execute(sitk.SmoothingRecursiveGaussian(refImg,0.25),
                         sitk.SmoothingRecursiveGaussian(inImg,0.25) )

    if method == ndregTranslation:
        idAffine = list(sitk.AffineTransform(inDimension).GetParameters())
        affine = idAffine[0:inDimension**2] + list(transform.GetOffset())
    else:
        affine = list(transform.GetMatrix()) + list(transform.GetTranslation())
    return affine

def imgAffineComposite(inImg, refImg, scale=1.0, useNearest=False, useMI=False, iterations=1000, inAffine=identityAffine,verbose=False, inMask=None, refMask=None, outDirPath=""):
    if outDirPath != "": outDirPath = dirMake(outDirPath)

    origInImg = inImg
    origInMask = inMask
    origRefMask = refMask

    #initilize using input affine
    compositeAffine = inAffine
    inImg = imgApplyAffine(origInImg, compositeAffine)
    if(inMask): inMask = imgApplyAffine(origInMask, compositeAffine, useNearest=True)

    if outDirPath != "":
        imgWrite(inImg, outDirPath+"0_initial/in.img")
        if(inMask): imgWrite(inMask, outDirPath+"0_initial/inMask.img")
        txtWriteList(compositeAffine, outDirPath+"0_initial/affine.txt")

    methodList = [ndregTranslation, ndregRigid, ndregAffine]
    methodNameList = ["translation", "rigid", "affine"]
    for (step, method) in enumerate(methodList):
        methodName = methodNameList[step]
        stepDirPath = outDirPath + str(step+1) + "_" + methodName + "/"
        if outDirPath != "": dirMake(stepDirPath)
        if(verbose): print("Step {0}:".format(methodName))

        affine = imgAffine(inImg, refImg, method=method, scale=scale, useNearest=useNearest, useMI=useMI, iterations=iterations, inMask=inMask, refMask=refMask, verbose=verbose)
        compositeAffine = affineApplyAffine(affine, compositeAffine)

        inImg = imgApplyAffine(origInImg, compositeAffine, size=refImg.GetSize())
        if(inMask): inMask = imgApplyAffine(origInMask, compositeAffine, size=refImg.GetSize(), useNearest=False)

        if outDirPath != "":
            imgWrite(inImg, stepDirPath+"in.img")
            if(inMask): imgWrite(inMask, stepDirPath+"inMask.img")
            txtWriteList(compositeAffine, stepDirPath+"affine.txt")

    # Write final results
    if outDirPath != "":
        txtWrite(compositeAffine, outDirPath+"affine.txt")
        imgWrite(inImg, outDirPath+"out.img")
        imgWrite(imgChecker(inImg, refImg), outDirPath+"checker.img")
    
    return compositeAffine    

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
        #imgWrite(imgChecker(inImg,refImg), outDirPath+"checker.img")
    
    if useTempDir: shutil.rmtree(outDirPath)
    return compositeField

def imgRegistration(inImg, refImg, scale=1.0, affineScale=1.0, lddmmScaleList=[1.0], lddmmAlphaList=[0.02], iterations=1000, useMI=False, Metric=0, useNearest=True, inAffine=identityAffine, padding=0, inMask=None, refMask=None, verbose=False, outDirPath=""):
    if outDirPath != "": outDirPath = dirMake(outDirPath)

    initialDirPath = outDirPath + "0_initial/"
    affineDirPath = outDirPath + "1_affine/"
    lddmmDirPath = outDirPath + "2_lddmm/"
    origInImg = inImg
    origRefImg = refImg

    # Resample and histogram match in and ref images
    refSpacing = refImg.GetSpacing()
    #print(inImg.GetSpacing())
    spacing = [x / scale for x in refSpacing]
    inImg = imgPad(imgResample(inImg, spacing, useNearest=useNearest), padding, useNearest)
    refImg = imgPad(imgResample(refImg, spacing, useNearest=useNearest), padding, useNearest)
    if(inMask): inMask = imgPad(imgResample(inMask, spacing, useNearest=True), padding, True)
    if(refMask): refMask = imgPad(imgResample(refMask, spacing, useNearest=True), padding, True)

    if not useMI: inImg = imgHM(inImg, refImg)
    initialInImg = inImg
    initialInMask = inMask
    initialRefMask = refMask
    if outDirPath != "":
        imgWrite(inImg, initialDirPath+"in.img")
        imgWrite(refImg, initialDirPath+"ref.img")
        if(inMask): imgWrite(inMask, initialDirPath+"inMask.img")
        if(refMask): imgWrite(refMask, initialDirPath+"refMask.img")

    if(verbose): print("Affine alignment")    
    affine = imgAffineComposite(inImg, refImg, scale=affineScale, useMI=useMI,iterations=1000, inAffine=inAffine, verbose=verbose, inMask=inMask, refMask=refMask, outDirPath=affineDirPath)
    affineField = affineToField(affine, refImg.GetSize(), refImg.GetSpacing())
    invAffine = affineInverse(affine)
    invAffineField = affineToField(invAffine, inImg.GetSize(), inImg.GetSpacing())
    inImg = imgApplyField(initialInImg, affineField, size=refImg.GetSize())
    if(inMask): inMask = imgApplyField(initialInMask, affineField, size=refImg.GetSize(), useNearest=True)
    if(refMask): refMask = imgApplyField(initialRefMask, affineField, size=refImg.GetSize(), useNearest=True)

    if outDirPath != "":
        imgWrite(inImg, affineDirPath+"in.img")
        if(inMask): imgWrite(inMask, affineDirPath+"inMask.img")
        if(refMask): imgWrite(refMask, affineDirPath+"refMask.img")

    # Deformably align in and ref images
    if(verbose): print("Deformable alignment")
    field= imgMetamorphosisComposite(inImg, refImg, alphaList=lddmmAlphaList,  MASGD=True, scaleList=lddmmScaleList, Metric=Metric, verbose=verbose, iterations=iterations,  outDirPath=lddmmDirPath)

    field = fieldApplyField(field, affineField)
    #invField = fieldApplyField(invAffineField, invField)
    inImg = imgApplyField(initialInImg, field, size=refImg.GetSize())

    if outDirPath != "":
        imgWrite(field, lddmmDirPath+"field.vtk")
        #imgWrite(invField, lddmmDirPath+"invField.vtk")
        imgWrite(inImg, lddmmDirPath+"in.img")    
        imgWrite(imgChecker(inImg, refImg), lddmmDirPath+"checker.img")

    # Remove padding from fields
    field = imgPad(field, -padding)
    #invField = imgPad(invField, -padding)

    if outDirPath != "":
        imgWrite(field, outDirPath+"field.vtk")
        #imgWrite(invField, outDirPath+"invField.vtk")

    return field

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

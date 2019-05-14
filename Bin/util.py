import SimpleITK as sitk
import os, math, sys, subprocess, tempfile, shutil
import os.path

def isNumber(variable):
    try:
        float(variable)
    except TypeError:
        return False
    return True

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
    textFile.close()

def dirMake(dirPath):
    if dirPath != "":
        if not os.path.exists(dirPath): os.makedirs(dirPath)
        return os.path.normpath(dirPath) + "/"
    else:
        return dirPath

def delFile(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)
    else:
        print('no such file:%s'%fileName)

def imgRead(path):
    """
    Alias for sitk.ReadImage
    """
    inImg = sitk.ReadImage(path)        
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

    # Reformat files to be compatible with CIS Software
    #ext = os.path.splitext(path)[1].lower()
    #if ext == ".vtk": vtkReformat(path, path)

def vtkReformat(inPath, outPath):
    """
    Reformats vtk file so that it can be read by CIS software.
    """
    # Get size of map
    inFile = open(inPath,"rb")
    lineList = inFile.readlines()
    for line in lineList:
        if line.lower().strip().startswith("dimensions"):
            size = map(int,line.split(" ")[1:dimension+1])
            break
    inFile.close()

    if dimension == 2: size += [0]

    outFile = open(outPath,"wb")
    for (i,line) in enumerate(lineList):
        if i == 1:
            newline = line.lstrip(line.rstrip("\n"))
            line = "lddmm 8 0 0 {0} {0} 0 0 {1} {1} 0 0 {2} {2}".format(size[2]-1, size[1]-1, size[0]-1) + newline
        outFile.write(line)

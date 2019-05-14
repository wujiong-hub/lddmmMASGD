#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import skimage
from skimage import  morphology
import scipy.ndimage.filters as filters
import math
import SimpleITK as sitk

def Image_orientation(inImg, inImg_orientation=None, refImg_orientation=None):
    if (inImg_orientation!=None and refImg_orientation!=None):
        inImg = imgReorient(inImg, inImg_orientation, refImg_orientation)
    return inImg

def Image_bias(inImg, refImg):
    mask_dilation_radius = 10 # voxels
    inImgMask = sitk.BinaryDilate(create_mask(inImg, use_triangle=True), mask_dilation_radius)
    refImgMak = sitk.BinaryDilate(create_mask(refImg, use_triangle=True), mask_dilation_radius)
    inImg = correct_bias_field(inImg, scale=0.5, mask=sitk.Cast(inImgMask,sitk.sitkUInt8), niters=[500,500,500,500])
    refImg = correct_bias_field(refImg, scale=0.5, mask=sitk.Cast(refImgMak,sitk.sitkUInt8), niters=[500,500,500,500])
    return inImg, refImg

def preprocess_Label(inLab, refLab, inImg_orientation=None, refImg_orientation=None):
    fixedSpacing=[1.0,1.0,1.0]
    inLab = imgResample(inLab, fixedSpacing, useNearest=True)
    refLab = imgResample(refLab, fixedSpacing, useNearest=True)
    if (inImg_orientation != None and refImg_orientation != None):
        inLab = imgReorient(inLab, inImg_orientation, refImg_orientation)
    return  inLab, refLab

def postprocess_Image(inImg, size, spacing):
    inImg = imgResample(inImg, size, spacing)
    return inImg

def postprocess_Label(inLab, size, spacing):
    inLab = imgResample(inLab, size, spacing, useNearest=True)
    return inLab


def create_mask(img, use_triangle=False):
    """Creates a mask of the image to separate brain from background using triangle or otsu thresholding. Otsu thresholding is the default.

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Image to compute the mask on.
    use_triangle : {bool}, optional
        Set to True if you want to use triangle thresholding. (the default is False, which results in Otsu thresholding)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Binary mask with 1s as the foreground and 0s as the background.
    """

    test_mask = None
    if use_triangle:
        test_mask = sitk.GetArrayFromImage(sitk.TriangleThreshold(img, 0, 1))
    else:
        test_mask = sitk.GetArrayFromImage(sitk.OtsuThreshold(img, 0, 1))
    eroded_im = morphology.opening(test_mask, selem=morphology.ball(2))
    connected_comp = skimage.measure.label(eroded_im)
    out = skimage.measure.regionprops(connected_comp)
    area_max = 0.0
    idx_max = 0
    for i in range(len(out)):
        if out[i].area > area_max:
            area_max = out[i].area
            idx_max = i+1
    connected_comp[ connected_comp != idx_max ] = 0
    mask = connected_comp
    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.CopyInformation(img)
    return mask_sitk

def correct_bias_field(img, mask=None, scale=0.25, niters=[50, 50, 50, 50]):
    """Correct bias field in image using the N4ITK algorithm (http://bit.ly/2oFwAun)

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input image with bias field.
    mask : {SimpleITK.SimpleITK.Image}, optional
        If used, the bias field will only be corrected within the mask. (the default is None, which results in the whole image being corrected.)
    scale : {float}, optional
        Scale at which to compute the bias correction. (the default is 0.25, which results in bias correction computed on an image downsampled to 1/4 of it's original size)
    niters : {list}, optional
        Number of iterations per resolution. Each additional entry in the list adds an additional resolution at which the bias is estimated. (the default is [50, 50, 50, 50] which results in 50 iterations per resolution at 4 resolutions)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Bias-corrected image that has the same size and spacing as the input image.
    """

     # do in case image has 0 intensities
    # add a small constant that depends on
    # distribution of intensities in the image
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)
    std = math.sqrt(stats.GetVariance())
    img_rescaled = sitk.Cast(img, sitk.sitkFloat32) + 0.1*std

    spacing = np.array(img_rescaled.GetSpacing())/scale
    img_ds = imgResample(img_rescaled, spacing=spacing)
    img_ds = sitk.Cast(img_ds, sitk.sitkFloat32)


    # Calculate bias
    if mask is None:
        mask = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8)+1
        mask.CopyInformation(img_ds)
    else:
        if type(mask) is not sitk.SimpleITK.Image:
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(img)
            mask = mask_sitk
        mask = imgResample(mask, spacing=spacing)

    img_ds_bc = sitk.N4BiasFieldCorrection(img_ds, mask, 0.001, niters)
    bias_ds = img_ds_bc / sitk.Cast(img_ds, img_ds_bc.GetPixelID())


    # Upsample bias
    bias = imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

    img_bc = sitk.Cast(img, sitk.sitkFloat32) * sitk.Cast(bias, sitk.sitkFloat32)
    return img_bc

def remove_grid_artifact(img, z_axis=1, sigma=10, mask=None):
    """Remove the gridding artifact from COLM images.

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input image.
    z_axis : {int}, optional
        An int indicating which axis is the z-axis. Can be 0, 1, or 2 (the default is 1, which  indicates the 2nd dimension is the z axis)
    sigma : {int}, optional
        The variance of the gaussian used to blur the image. Larger sigma means more grid correction but stronger edge artifacts. (the default is 10, which empirically works well our data at 50 um)
    mask : {SimpleITK.SimpleITK.Image}, optional
        An image with 1s representing the foreground (brain) and 0s representing the background. (the default is None, which will use otsu thresholding to create the brain mask.)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Input image with grid artifact removed.
    """

    if mask == None: mask = sitk.GetArrayFromImage(sitk.OtsuThreshold(img))
    img_np = sitk.GetArrayFromImage(img)
    # create masked array
    out = np.ma.array(img_np, mask=mask)
    # compute masked average
    mean_z = np.ma.average(out, axis=z_axis)
    #stdev = math.sqrt(np.var(mean_z))
    #small_factor = 0.1
    bias_z_slice = filters.gaussian_filter(mean_z, sigma)/(mean_z)
    bias_z_img = np.expand_dims(bias_z_slice, z_axis)
    test = np.repeat(bias_z_img, img_np.shape[z_axis], axis=z_axis)
    img_c = img_np * (test * np.abs(mask - 1))
    img_c[ np.isnan(img_c) ] = 0.0
    img_c_sitk = sitk.GetImageFromArray(img_c)
    img_c_sitk.SetSpacing(img.GetSpacing())
    return img_c_sitk

def imgReorient(img, in_orient, out_orient):
    """Reorients input image to match out_orient.

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input 3D image.
    in_orient : {str}
        3-letter string indicating orientation of brain.
    out_orient : {str}
        3-letter string indicating desired orientation of input.

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Reoriented input image.
    """

    dimension = img.GetDimension()
    if (len(in_orient) != dimension):
        raise Exception(
            "in_orient must be a string of length {0}.".format(dimension))
    if (len(out_orient) != dimension):
        raise Exception(
            "out_orient must be a string of length {0}.".format(dimension))
    in_orient = str(in_orient).lower()
    out_orient = str(out_orient).lower()

    inDirection = ""
    outDirection = ""
    orientToDirection = {"r": "r", "l": "r",
                         "s": "s", "i": "s", "a": "a", "p": "a"}
    for i in range(dimension):
        try:
            inDirection += orientToDirection[in_orient[i]]
        except BaseException:
            raise Exception("in_orient \'{0}\' is invalid.".format(in_orient))

        try:
            outDirection += orientToDirection[out_orient[i]]
        except BaseException:
            raise Exception("out_orient \'{0}\' is invalid.".format(out_orient))

    if len(set(inDirection)) != dimension:
        raise Exception(
            "in_orient \'{0}\' is invalid.".format(in_orient))
    if len(set(outDirection)) != dimension:
        raise Exception(
            "out_orient \'{0}\' is invalid.".format(out_orient))

    order = []
    flip = []
    for i in range(dimension):
        j = inDirection.find(outDirection[i])
        order += [j]
        flip += [in_orient[j] != out_orient[i]]

    print(order)
    print(flip)

    out_img = sitk.PermuteAxesImageFilter().Execute(img, order)
    out_img = sitk.FlipImageFilter().Execute(out_img, flip, False)
    return out_img

def imgResample(img, spacing, size=[], useNearest=False, origin=None, outsideValue=0):
    """Resample image to certain spacing and size.

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input 3D image.
    spacing : {list}
        List of length 3 indicating the voxel spacing as [x, y, z]
    size : {list}, optional
        List of length 3 indicating the number of voxels per dim [x, y, z] (the default is [], which will use compute the appropriate size based on the spacing.)
    useNearest : {bool}, optional
        If True use nearest neighbor interpolation. (the default is False, which will use linear interpolation.)
    origin : {list}, optional
        The location in physical space representing the [0,0,0] voxel in the input image. (the default is [0,0,0])
    outsideValue : {int}, optional
        value used to pad are outside image (the default is 0)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Resampled input image.
    """

    if origin is None: origin = [0]*3
    if len(spacing) != img.GetDimension():
        raise Exception(
            "len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size == []:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [int(math.ceil(inSize[i] * (inSpacing[i] / spacing[i])))
                for i in range(img.GetDimension())]
    else:
        if len(size) != img.GetDimension():
            raise Exception(
                "len(size) != " + str(img.GetDimension()))

    # Resample input image
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    identityTransform = sitk.Transform()
    identityDirection = list(
        sitk.AffineTransform(
            img.GetDimension()).GetMatrix())

    return sitk.Resample(
        img,
        size,
        identityTransform,
        interpolator,
        origin,
        spacing,
        img.GetDirection(),
        outsideValue)

def normalize(img, percentile=0.99):
    if percentile < 0.0 or percentile > 1.0:
        raise Exception("Percentile should be between 0.0 and 1.0")

    #Accept ndarray images or sitk images
    if type(img) is np.ndarray:
        sitk_img = sitk.GetImageFromArray(img)
    else:
        sitk_img = img

    #just taking code from ndreg.py....
    (values, bins) = np.histogram(sitk.GetArrayFromImage(sitk_img), bins=255)
    cumValues = np.cumsum(values).astype(float)
    cumValues = (cumValues - cumValues.min()) / cumValues.ptp()

    index = np.argmax(cumValues > percentile) - 1
    max_val = bins[index]

    return sitk.Clamp(sitk_img, upperBound=max_val) / max_val

def downsample_and_reorient(atlas, target, atlas_orient, target_orient, spacing, size=[], set_origin=True, dv_atlas=0.0, dv_target=0.0):
    """
    make sure img1 is the source and img2 is the target.
    iamges will be resampled to match the coordinate system of img2.
    """
    target_r = imgReorient(target, target_orient, atlas_orient)
    size_atlas = atlas.GetSize()
    size_target = target_r.GetSize()
    dims_atlas = np.array(size_atlas)*np.array(atlas.GetSpacing())
    dims_target = np.array(size_target)*np.array(target_r.GetSpacing())
    max_size_per_dim = [max(dims_atlas[i], dims_target[i]) for i in range(len(dims_atlas))]
    vox_sizes = [int(i) for i in (np.array(max_size_per_dim) / spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetSize(vox_sizes)
    resampler.SetOutputSpacing([spacing]*3)
    resampler.SetDefaultPixelValue(dv_target)

    out_target = resampler.Execute(target_r)
    resampler.SetDefaultPixelValue(dv_atlas)
    out_atlas = resampler.Execute(atlas)
    return out_atlas, out_target

def imgHM(img, ref_img, numMatchPoints=64, numBins=256):
    """Performs histogram matching on two images.

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Image on which histogram matching is performed.
    ref_img : {SimpleITK.SimpleITK.Image}
        reference image for histogram matching.
    numMatchPoints : {int}, optional
        number of quantile values to be matched. (the default is 64)
    numBins : {int}, optional
        Number of bins used in  computation of the histogram(the default is 256)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Input image histgram-matched to the reference image.
    """

    img = sitk.Cast(img, ref_img.GetPixelID())
    return sitk.HistogramMatchingImageFilter().Execute(
        img, ref_img, numBins, numMatchPoints, False)


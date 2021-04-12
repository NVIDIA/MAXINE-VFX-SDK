/*###############################################################################
#
# Copyright (c) 2020 NVIDIA Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
###############################################################################*/

#ifndef __BATCH_UTILITIES__
#define __BATCH_UTILITIES__

#include "nvCVImage.h"


//! Allocate a batch buffer.
//! \note All of the arguments are identical to that of NvCVImage_Alloc plus the batchSize.
//! \param[out] im        the image to initialize.
//! \param[in]  batchSize the number i=of images in the batch.
//! \param[in]  width     the desired width  of each image, in pixels.
//! \param[in]  height    the desired height of each image, in pixels.
//! \param[in]  format    the format of the pixels.
//! \param[in]  type      the type of the components of the pixels.
//! \param[in]  layout    One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.
//! \param[in]  memSpace  Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }
//! \param[in]  alignment row byte alignment. Choose 0 or a power of 2.
//!                       1: yields no gap whatsoever between scanlines;
//!                       0: default alignment: 4 on CPU, and cudaMallocPitch's choice on GPU.
//!                       Other common values are 16 or 32 for cache line size, 32 for texture alignment.
//! \return NVCV_SUCCESS         if the operation was successful.
//! \return NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
//! \return NVCV_ERR_MEMORY      if there is not enough memory to allocate the buffer.
//! \note   this simply multiplies height by batchSize and calls NvCVImage_Alloc().
NvCV_Status AllocateBatchBuffer(NvCVImage* im, unsigned batchSize, unsigned width, unsigned height,
  NvCVImage_PixelFormat format, NvCVImage_ComponentType type, unsigned layout, unsigned memSpace, unsigned alignment);

//! Initialize an image descriptor for the Nth image in a batch.
//! \param[in]  n       the index of the desired image in the batch.
//! \param[in]  height  the height of the image
//! \param[in]  full    the batch image, or the 0th image in the batch.
//! \param[out] view    the image descriptor to be initialized to a view of the nth image in the batch.
//! \return     a pointer to the nth image view, facilitating the use of NthImage() inline as an argument to a function.
//! \note       NvCVImage nth; NvVFX_SetImage(effect, NVVFX_INPUT_IMAGE, NthImage(0, height, batchIn, &nth));
//!             is typically used to set the input image for a batch operation; similarly for output.
NvCVImage* NthImage(unsigned n, unsigned height, NvCVImage* full, NvCVImage* view);

//! Compute the byte offset between one image in a batch and the next.
//! \param[in]  im  the image to be measured.
//! \return the increment from one image to the next in a batch.
//! \note this will be negative if the pitch is negative.
int ComputeImageBytes(const NvCVImage* im);

//! Transfer To the Nth Image in a Batched Image.
//! \param[in]  n         the index of the batch image to modify.
//! \param[in]  src       the source image.
//! \param[in]  dstBatch  the batch destination image.
//! \param[in]  scale     the pixel scale factor.
//! \param[in]  stream    the CUDA stream on which to perform the transfer.
//! \param[in]  tmp       the stage buffer (can be NULL, but can affect performance if needed).
//! \return NVCV_SUCCESS if the operation was successful.
NvCV_Status TransferToNthImage(
  unsigned n, const NvCVImage* src, NvCVImage* dstBatch, float scale, struct CUstream_st* stream, NvCVImage* tmp);

//! Transfer From the Nth Image in a Batched Image.
//! \param[in]  n         the index of the batch image to read.
//! \param[in]  srcBatch  the batch source image.
//! \param[in]  dst       the destination image.
//! \param[in]  scale     the pixel scale factor.
//! \param[in]  stream    the CUDA stream on which to perform the transfer.
//! \param[in]  tmp       the stage buffer (can be NULL, but can affect performance if needed).
//! \return NVCV_SUCCESS if the operation was successful.
NvCV_Status TransferFromNthImage(
  unsigned n, const NvCVImage* srcBatch, NvCVImage* dst, float scale, struct CUstream_st* stream, NvCVImage* tmp);

//! Transfer from a list of source images to a batch image.
//! We use an array of image pointers rather than an array of images
//! in order to more easily accommodate dynamically-changing batches.
//! \param[in]  batchSize the number of source images to be transferred to the batch image.
//! \param[in]  srcArray  array of pointers to the source images.
//! \param[out] dstBatch  the batch destination image.
//! \param[in]  scale     the pixel scale factor.
//! \param[in]  stream    the CUDA stream.
//! \param[in]  tmp       the stage buffer (can be NULL, but can affect performance if needed).
//! \return NVCV_SUCCESS  if the operation was successful.
NvCV_Status TransferToBatchImage(
  unsigned batchSize, const NvCVImage** srcArray, NvCVImage* dstBatch, float scale, struct CUstream_st* stream, NvCVImage* tmp);

//! Transfer from a batch image to a list of destination images.
//! We use an array of image pointers rather than an array of images
//! in order to more easily accommodate dynamically-changing batches.
//! \param[in]  batchSize the number of destination images to be transferred from the batch image.
//! \param[in]  srcBatch  the batch source image.
//! \param[out] dstArray  array of pointers to the source images.
//! \param[in]  scale     the pixel scale factor.
//! \param[in]  stream    the CUDA stream.
//! \param[in]  tmp       the stage buffer (can be NULL, but can affect performance if needed).
//! \return NVCV_SUCCESS  if the operation was successful.
NvCV_Status TransferFromBatchImage(
  unsigned batchSize, const NvCVImage* srcBatch, NvCVImage** dstArray, float scale, struct CUstream_st* stream, NvCVImage* tmp);

//! Transfer all images in a batch to another compatible batch of images.
//! \param[in]  srcBatch  the batch source image.
//! \param[out] dstBatch  the batch destination image.
//! \param[in]  imHeight  the height of each image in the batch.
//! \param[in]  batchSize the number of images in the batch.
//! \param[in]  scale     the pixel scale factor.
//! \param[in]  stream    the CUDA stream.
//! \return NVCV_SUCCESS  if the operation was successful.
NvCV_Status TransferBatchImage(const NvCVImage* srcBatch, NvCVImage* dstBatch,
  unsigned imHeight, unsigned batchSize, float scale, struct CUstream_st* stream);


#endif // __BATCH_UTILITIES__

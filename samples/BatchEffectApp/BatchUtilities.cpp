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

#include "BatchUtilities.h"


/********************************************************************************
 * AllocateBatchBuffer
 ********************************************************************************/

NvCV_Status AllocateBatchBuffer(NvCVImage *im, unsigned batchSize, unsigned width, unsigned height, NvCVImage_PixelFormat format,
  NvCVImage_ComponentType type, unsigned layout, unsigned memSpace, unsigned alignment) {
  return NvCVImage_Alloc(im, width, height * batchSize, format, type, layout, memSpace, alignment);
}


/********************************************************************************
 * NthImage
 ********************************************************************************/

NvCVImage* NthImage(unsigned n, unsigned height, NvCVImage* full, NvCVImage* view) {
  unsigned y = height;
  if        (NVCV_PLANAR &  full->planar) {    // if not any of the chunky formats
    if      (NVCV_PLANAR == full->planar)      y *= full->numComponents;
    else if (NVCV_YUV444 == full->pixelFormat) y *= 3;
    else if (NVCV_YUV422 == full->pixelFormat) y *= 2;
    else if (NVCV_YUV420 == full->pixelFormat) y = y * 3 / 2;
    else                                       y = 0;
  }
  NvCVImage_InitView(view, full, 0, y * n, full->width, height);
  return view;
}


/********************************************************************************
 * ComputeImageBytes
 ********************************************************************************/

int ComputeImageBytes(const NvCVImage* im) {
  int imageBytes = im->pitch * (int)im->height;  // Correct for all chunky formats
  if        (NVCV_PLANAR &  im->planar) {    // if not any of the chunky formats
    if      (NVCV_PLANAR == im->planar)      imageBytes *= (int)im->numComponents;
    else if (NVCV_YUV422 == im->pixelFormat) imageBytes *= 2;
    else if (NVCV_YUV420 == im->pixelFormat) imageBytes = imageBytes * 3 / 2;
    else                                     imageBytes = 0;
  }
  return imageBytes;
}


/********************************************************************************
 * TransferToNthImage
 ********************************************************************************/

NvCV_Status TransferToNthImage(
  unsigned n, const NvCVImage* src, NvCVImage* dstBatch, float scale, struct CUstream_st* stream, NvCVImage* tmp) {
  NvCVImage nth;
  return NvCVImage_Transfer(src, NthImage(n, src->height, dstBatch, &nth), scale, stream, tmp);
}


/********************************************************************************
 * TransferFromNthImage
 ********************************************************************************/

NvCV_Status TransferFromNthImage(
  unsigned n, const NvCVImage* srcBatch, NvCVImage* dst, float scale, struct CUstream_st* stream, NvCVImage* tmp) {
  NvCVImage nth;
  return NvCVImage_Transfer(NthImage(n, dst->height, const_cast<NvCVImage*>(srcBatch), &nth), dst, scale, stream, tmp);
}


/********************************************************************************
 * TransferToBatchImage
 * This illustrates the use of the pixel offset method, but the Nth image method could be used instead.
 ********************************************************************************/

NvCV_Status TransferToBatchImage(
  unsigned batchSize, const NvCVImage** srcArray, NvCVImage* dstBatch, float scale, struct CUstream_st* stream, NvCVImage* tmp) {
  NvCV_Status err = NVCV_SUCCESS;
  NvCVImage nth;
  (void)NthImage(0, (**srcArray).height, dstBatch, &nth);
  int nextDst = ComputeImageBytes(&nth);
  for (; batchSize--; ++srcArray, nth.pixels = (void*)((char*)nth.pixels + nextDst))
    if (NVCV_SUCCESS != (err = NvCVImage_Transfer(*srcArray, &nth, scale, stream, tmp)))
      break;
  return err;
}


/********************************************************************************
 * TransferFromBatchImage
 * This illustrates the use of the pixel offset method, but the Nth image method could be used instead.
 ********************************************************************************/

NvCV_Status TransferFromBatchImage(
  unsigned batchSize, const NvCVImage* srcBatch, NvCVImage** dstArray, float scale, struct CUstream_st* stream, NvCVImage* tmp) {
  NvCV_Status err = NVCV_SUCCESS;
  NvCVImage nth;
  (void)NthImage(0, (**dstArray).height, const_cast<NvCVImage*>(srcBatch), &nth);
  int nextSrc = ComputeImageBytes(&nth);
  for (; batchSize--; nth.pixels = (void*)((char*)nth.pixels + nextSrc), ++dstArray)
    if (NVCV_SUCCESS != (err = NvCVImage_Transfer(&nth, *dstArray, scale, stream, tmp)))
      break;
  return err;
}


/********************************************************************************
 * TransferBatchImage
 ********************************************************************************/

NvCV_Status TransferBatchImage(const NvCVImage *srcBatch, NvCVImage *dstBatch,
        unsigned imHeight, unsigned batchSize, float scale, struct CUstream_st *stream) {
  NvCV_Status err = NVCV_SUCCESS;
  NvCVImage   tmp;

  if ((!(srcBatch->planar & NVCV_PLANAR) && !(dstBatch->planar & NVCV_PLANAR))  // both chunky
    || (srcBatch->planar == NVCV_PLANAR && dstBatch->planar == NVCV_PLANAR && srcBatch->pixelFormat == dstBatch->pixelFormat)
  ) {     // This is a fast transfer
    err = NvCVImage_Transfer(srcBatch, dstBatch, scale, stream, &tmp);
  }
  else {  // This is guaranteed to be safe for all transfers
    NvCVImage subSrc, subDst;
    int       nextSrc, nextDst, n;
    NvCVImage_InitView(&subSrc, const_cast<NvCVImage*>(srcBatch), 0, 0, srcBatch->width, imHeight);
    NvCVImage_InitView(&subDst, dstBatch, 0, 0, dstBatch->width, imHeight);
    nextSrc = ComputeImageBytes(&subSrc);
    nextDst = ComputeImageBytes(&subDst);
    for (n = batchSize; n--; subSrc.pixels = (char*)subSrc.pixels + nextSrc,
                             subDst.pixels = (char*)subDst.pixels + nextDst)
      if (NVCV_SUCCESS != (err = NvCVImage_Transfer(&subSrc, &subDst, scale, stream, &tmp)))
        break;
  }
  return err;
}

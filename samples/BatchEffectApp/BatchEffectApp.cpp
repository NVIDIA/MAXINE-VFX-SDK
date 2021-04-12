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

#include <stdio.h>
#include <string.h>

#include <string>

#include "BatchUtilities.h"
#include "nvCVOpenCV.h"
#include "nvVideoEffects.h"
#include "opencv2/opencv.hpp"

#ifdef _MSC_VER
  #define strcasecmp _stricmp
#endif // _MSC_VER

#define BAIL_IF_ERR(err)            do { if (0 != (err)) {                      goto bail; } } while(0)
#define BAIL_IF_NULL(x, err, code)  do { if ((void*)(x) == NULL)  { err = code; goto bail; } } while(0)
#define BAIL_IF_FALSE(x, err, code) do { if (!(x))                { err = code; goto bail; } } while(0)
#define BAIL(err, code)             do {                            err = code; goto bail;   } while(0)


bool                      FLAG_verbose        = false;
float                     FLAG_strength       = 0.f,
                          FLAG_scale          = 1.0;
int                       FLAG_mode           = 0,
                          FLAG_resolution     = 0;
std::string               FLAG_outFile,
                          FLAG_modelDir,
                          FLAG_effect;
std::vector<const char*>  FLAG_inFiles;

// Set this when using OTA Updates
// This path is used by nvVideoEffectsProxy.cpp to load the SDK dll
// when using  OTA Updates
char *g_nvVFXSDKPath = NULL;

static bool GetFlagArgVal(const char *flag, const char *arg, const char **val) {
  if (*arg != '-')
    return false;
  while (*++arg == '-')
    continue;
  const char *s = strchr(arg, '=');
  if (s == NULL)  {
    if (strcmp(flag, arg) != 0)
      return false;
    *val = NULL;
    return true;
  }
  size_t n = s - arg;
  if ((strlen(flag) != n) || (strncmp(flag, arg, n) != 0))
    return false;
  *val = s + 1;
  return true;
}

static bool GetFlagArgVal(const char *flag, const char *arg, std::string *val) {
  const char *valStr;
  if (!GetFlagArgVal(flag, arg, &valStr))
    return false;
  val->assign(valStr ? valStr : "");
  return true;
}

static bool GetFlagArgVal(const char *flag, const char *arg, bool *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) {
    *val = (valStr == NULL ||
      strcasecmp(valStr, "true") == 0 ||
      strcasecmp(valStr, "on")   == 0 ||
      strcasecmp(valStr, "yes")  == 0 ||
      strcasecmp(valStr, "1")    == 0
      );
  }
  return success;
}

static bool GetFlagArgVal(const char *flag, const char *arg, float *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success)
    *val = strtof(valStr, NULL);
  return success;
}

static bool GetFlagArgVal(const char *flag, const char *arg, long *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success)
    *val = strtol(valStr, NULL, 10);
  return success;
}

static bool GetFlagArgVal(const char *flag, const char *arg, int *val) {
  long longVal;
  bool success = GetFlagArgVal(flag, arg, &longVal);
  if (success)
    *val = (int)longVal;
  return success;
}

static void Usage() {
  printf(
    "BatchEffectApp [flags ...] inFile1 [ inFileN ...]\n"
    "  where flags is:\n"
    "  --out_file=<path>     output image files to be written, default \"BatchOut_%%02u.png\"\n"
    "  --effect=<effect>     the effect to apply\n"
    "  --strength=<value>    strength of an effect, 0 or 1 for super res and artifact reduction,\n"
    "                        and [0.0, 1.0] for upscaling\n"
    "  --scale=<scale>       scale factor to be applied: 1.5, 2, 3, maybe 1.3333333\n"
    "  --resolution=<height> the desired height (either --scale or --resolution may be used)\n"
    "  --mode=<mode>         mode 0 or 1\n"
    "  --model_dir=<path>    the path to the directory that contains the models\n"
    "  --verbose             verbose output\n"
    "  and inFile1 ... are identically sized image files, e.g. png, jpg\n"
  );

  const char* cStr;
  NvCV_Status err = NvVFX_GetString(nullptr, NVVFX_INFO, &cStr);
  if (NVCV_SUCCESS != err)
    printf("Cannot get effects: %s\n", NvCV_GetErrorStringFromCode(err));
  printf("where effects are:\n%s", cStr);
}

static int ParseMyArgs(int argc, char **argv) {
  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char *arg = *argv;
    if (arg[0] == '-') {
      if (arg[1] == '-') {                                      // double-dash
        if (GetFlagArgVal("verbose",    arg, &FLAG_verbose)   ||
            GetFlagArgVal("effect",     arg, &FLAG_effect)    ||
            GetFlagArgVal("strength",   arg, &FLAG_strength)  ||
            GetFlagArgVal("scale",      arg, &FLAG_scale)     ||
            GetFlagArgVal("mode",       arg, &FLAG_mode)      ||
            GetFlagArgVal("model_dir",  arg, &FLAG_modelDir)  ||
            GetFlagArgVal("out_file",   arg, &FLAG_outFile)
        ) {
          continue;
        } else if (GetFlagArgVal("help", arg, &help)) {         // --help
          Usage();
          errs = 1;
        }
      }
      else {                                                    // single dash
        for (++arg; *arg; ++arg) {
          if (*arg == 'v') {
            FLAG_verbose = true;
          } else {
            printf("Unknown flag ignored: \"-%c\"\n", *arg);
          }
        }
        continue;
      }
    }
    else {                                                      // no dash
      FLAG_inFiles.push_back(arg);
    }
  }
  return errs;
}


class App {
public:
  NvVFX_Handle  _eff;
  NvCVImage     _src, _dst, _stg;
  CUstream      _stream;
  unsigned      _batchSize;

  App() : _eff(nullptr), _stream(0), _batchSize(0) {}
  ~App() { NvVFX_DestroyEffect(_eff); if (_stream) NvVFX_CudaStreamDestroy(_stream); }

  NvCV_Status init(const char* effectName, unsigned batchSize, const NvCVImage *src) {
    NvCV_Status err = NVCV_ERR_UNIMPLEMENTED;
    unsigned    dw, dh;

    if (FLAG_resolution) {
      dw = FLAG_resolution * src->width / src->height,  // No rounding
      dh = FLAG_resolution;
    }
    else {
      dw = lroundf(src->width  * FLAG_scale),
      dh = lroundf(src->height * FLAG_scale);
    }

    _batchSize = batchSize;
    BAIL_IF_ERR(err = NvVFX_CreateEffect(effectName, &_eff));

    if (!strcmp(effectName, NVVFX_FX_TRANSFER)) {
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_src, _batchSize, src->width, src->height, NVCV_RGB, NVCV_U8, NVCV_CHUNKY, NVCV_CUDA, 0));
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_dst, _batchSize, src->width, src->height, NVCV_RGB, NVCV_U8, NVCV_CHUNKY, NVCV_CUDA, 0));
    }
#ifdef NVVFX_FX_SR_UPSCALE
    else if (!strcmp(effectName, NVVFX_FX_SR_UPSCALE)) {
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_src, _batchSize, src->width, src->height, NVCV_RGBA, NVCV_U8, NVCV_CHUNKY, NVCV_CUDA, 32)); // n*32, n>=0
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_dst, _batchSize, dw,         dh,          NVCV_RGBA, NVCV_U8, NVCV_CHUNKY, NVCV_CUDA, 32));
    }
#endif // NVVFX_FX_SR_UPSCALE
#ifdef NVVFX_FX_GREEN_SCREEN
    else if (!strcmp(effectName, NVVFX_FX_GREEN_SCREEN)) {
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_src, _batchSize, src->width, src->height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CUDA, 1));
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_dst, _batchSize, src->width, src->height, NVCV_Y,   NVCV_U8, NVCV_CHUNKY, NVCV_CUDA, 1));
      BAIL_IF_ERR(err = NvVFX_SetString(_eff, NVVFX_MODEL_DIRECTORY, FLAG_modelDir.c_str()));
      BAIL_IF_ERR(err = NvVFX_SetU32(_eff, NVVFX_MODE, FLAG_mode));
    }
#endif // NVVFX_FX_GREEN_SCREEN
#ifdef NVVFX_FX_ARTIFACT_REDUCTION
    else if (!strcmp(effectName, NVVFX_FX_ARTIFACT_REDUCTION)) {
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_src, _batchSize, src->width, src->height, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_CUDA, 1));
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_dst, _batchSize, src->width, src->height, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_CUDA, 1));
      BAIL_IF_ERR(err = NvVFX_SetString(_eff, NVVFX_MODEL_DIRECTORY, FLAG_modelDir.c_str()));
      BAIL_IF_ERR(err = NvVFX_SetU32(_eff, NVVFX_STRENGTH, (unsigned)FLAG_strength));
    }
#endif // NVVFX_FX_ARTIFACT_REDUCTION
#ifdef NVVFX_FX_SUPER_RES
    else if (!strcmp(effectName, NVVFX_FX_SUPER_RES)) {
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_src, _batchSize, src->width, src->height, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_CUDA, 1));
      BAIL_IF_ERR(err = AllocateBatchBuffer(&_dst, _batchSize, dw,         dh,          NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_CUDA, 1));
      BAIL_IF_ERR(err = NvVFX_SetString(_eff, NVVFX_MODEL_DIRECTORY, FLAG_modelDir.c_str()));
      BAIL_IF_ERR(err = NvVFX_SetU32(_eff, NVVFX_STRENGTH, (unsigned)FLAG_strength));
    }
#endif // NVVFX_FX_SUPER_RES
    else {
      BAIL(err, NVCV_ERR_UNIMPLEMENTED);
    }

    { // Set common parameters.
      NvCVImage nth;
      BAIL_IF_ERR(err = NvVFX_SetImage(_eff, NVVFX_INPUT_IMAGE,  NthImage(0, src->height,              &_src, &nth)));  // Set the first of the batched images in ...
      BAIL_IF_ERR(err = NvVFX_SetImage(_eff, NVVFX_OUTPUT_IMAGE, NthImage(0, _dst.height / _batchSize, &_dst, &nth)));  // ... and out
      BAIL_IF_ERR(err = NvVFX_CudaStreamCreate(&_stream));
      BAIL_IF_ERR(err = NvVFX_SetCudaStream(_eff, NVVFX_CUDA_STREAM, _stream));

      // The batch size parameter is interpreted at two times:
      // (1) during Load(), an appropriate batch-size model is chosen and loaded;
      // (2) during Run(), the specified number of images in the batch are processed.
      // The optimum throughput results from submitting a batch which is an integral multiple of the batched model
      // chosen in Load().
      //
      // To request a particular batch-sized model, set the batch size before calling Load(),
      // then get the batch size afterward to find out what batch-size model was chosen. If you do not specify the
      // desired batchSize before calling Load(), it will choose the batchSize=1 model, since that is the default
      // value for batchSize.
      //
      // After calling Load(), you can subsequently change the batch size to any number, even larger or smaller
      // than the batch size of the chosen model. If a larger  batch size is chosen, smaller batches are submitted
      // until the entire larger batch has been processed. In any event, the batch size should be set at least twice:
      // once before Load() and once before the initial Run(). In many server applications, it is expected that
      // the batch size is changing constantly as some videos complete and other are added, so setting the batchSize
      // before every Run() call would be typical.
      unsigned gotBatch;
      BAIL_IF_ERR(err = NvVFX_SetU32(_eff, NVVFX_MODEL_BATCH, _batchSize)); // Try to choose a model tuned to this batch size
      err = NvVFX_Load(_eff);                                               // This will load a new batched model -- a weighty process
      if (!(NVCV_SUCCESS == err || NVCV_ERR_MODELSUBSTITUTION == err)) goto bail;
      BAIL_IF_ERR(err = NvVFX_GetU32(_eff, NVVFX_MODEL_BATCH, &gotBatch));  // This tells us the batch size of the chosen model
      if (FLAG_verbose && gotBatch != _batchSize) {
        printf("Effect %s has no batch=%u model; processing in multiple batches of size %u%s instead\n",
            effectName, _batchSize, gotBatch, (gotBatch > 1 ? " or less" : ""));
        BAIL_IF_ERR(err = NvVFX_SetU32(_eff, NVVFX_BATCH_SIZE, _batchSize));  // This is lightweight, and usually done each Run
      }
    }

  bail:
    return err;
  }
};


NvCV_Status BatchProcessImages(const char* effectName, const std::vector<const char*>& srcImages, const char *outfilePattern) {
  NvCV_Status err       = NVCV_SUCCESS;
  unsigned    batchSize = (unsigned)srcImages.size();
  App         app;
  cv::Mat     ocv;
  NvCVImage   nvx;
  unsigned    srcWidth, srcHeight, dstHeight, i;

  // Read in the first image, to determine the resolution for init()
  BAIL_IF_FALSE(srcImages.size() > 0, err, NVCV_ERR_MISSINGINPUT);
  ocv = cv::imread(srcImages[0]);
  if (!ocv.data) {
    printf("Cannot read image file \"%s\"\n", srcImages[0]);
    BAIL(err, NVCV_ERR_READ);
  }
  NVWrapperForCVMat(&ocv, &nvx);
  srcWidth  = nvx.width;
  srcHeight = nvx.height;
  BAIL_IF_ERR(err = app.init(effectName, batchSize, &nvx)); // Init effect and buffers

  // Transfer the first image to the batch src.
  // Note, in all transfers, the scale factor only applies to floating-point pixels.
  BAIL_IF_ERR(err = TransferToNthImage(0, &nvx, &app._src, 1.f/255.f, app._stream, &app._stg));
  ocv.release();

  // Read the remaining images and transfer to the batch src
  for (i = 1; i < batchSize; ++i) {
    ocv = cv::imread(srcImages[i]);
    if (!ocv.data) {
      printf("Cannot read image file \"%s\"\n", srcImages[i]);
      BAIL(err, NVCV_ERR_READ);
    }
    NVWrapperForCVMat(&ocv, &nvx);
    if (!(nvx.width == srcWidth && nvx.height == srcHeight)) {
      printf("Input image file \"%s\" %ux%u does not match %ux%u\n", srcImages[i], nvx.width, nvx.height, srcWidth, srcHeight);
      BAIL(err, NVCV_ERR_MISMATCH);
    }
    BAIL_IF_ERR(err = TransferToNthImage(i, &nvx, &app._src, 1.f / 255.f, app._stream, &app._stg));
    ocv.release();
  }

  // Run batch
  BAIL_IF_ERR(err = NvVFX_SetU32(app._eff, NVVFX_BATCH_SIZE, (unsigned)srcImages.size()));  // The batchSize can change every Run
  BAIL_IF_ERR(err = NvVFX_Run(app._eff, 0));

  // Retrieve and write images
  dstHeight = app._dst.height / batchSize;
  BAIL_IF_ERR(err = NvCVImage_Alloc(&nvx, app._dst.width, dstHeight, ((app._dst.numComponents == 1) ? NVCV_Y : NVCV_BGR), NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 0));
  CVWrapperForNvCVImage(&nvx, &ocv);
  for (i = 0; i < batchSize; ++i) {
    char fileName[1024];
    snprintf(fileName, sizeof(fileName), outfilePattern, i);
    BAIL_IF_ERR(err = TransferFromNthImage(i, &app._dst, &nvx, 255.f, app._stream, &app._stg));
    if (!cv::imwrite(fileName, ocv)) {
      printf("Cannot write image file \"%s\"\n", fileName);
      BAIL(err, NVCV_ERR_WRITE);
    }
  }
  // NvCVImage_Dealloc() is called in the destructors

bail:
  return err;
}


int main(int argc, char** argv) {
  int         nErrs;
  NvCV_Status vfxErr;

  nErrs = ParseMyArgs(argc, argv);
  if (nErrs)
    return nErrs;

  if (FLAG_outFile.empty())
    FLAG_outFile = "BatchOut_%02u.png";
  else if (std::string::npos == FLAG_outFile.find_first_of('%'))
    FLAG_outFile.insert(FLAG_outFile.size() - 4, "_%02u");  // assuming .xxx, i.e. .jpg, .png

  vfxErr = BatchProcessImages(FLAG_effect.c_str(), FLAG_inFiles, FLAG_outFile.c_str());
  if (NVCV_SUCCESS != vfxErr) {
    printf("Error: %s\n", NvCV_GetErrorStringFromCode(vfxErr));
    nErrs = (int)vfxErr;
  }

  return nErrs;
}

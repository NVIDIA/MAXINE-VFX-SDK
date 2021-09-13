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
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <string>
#include <iostream>

#include "nvCVOpenCV.h"
#include "nvVideoEffects.h"
#include "opencv2/opencv.hpp"


#ifdef _MSC_VER
  #define strcasecmp _stricmp
  #include <Windows.h>
#else // !_MSC_VER
  #include <sys/stat.h>
#endif // _MSC_VER

#define BAIL_IF_ERR(err)                    do { if (0 != (err)) {                      goto bail; } } while(0)
#define BAIL_IF_NULL(x, err, code)          do { if ((void*)(x) == NULL)  { err = code; goto bail; } } while(0)
#define NVCV_ERR_HELP 411

#ifdef _WIN32
  #define DEFAULT_CODEC "avc1"
#else // !_WIN32
  #define DEFAULT_CODEC "H264"
#endif // _WIN32


bool        FLAG_debug          = false,
            FLAG_verbose        = false,
            FLAG_show           = false,
            FLAG_progress       = false,
            FLAG_webcam         = false;
float       FLAG_strength       = 0.f;
int         FLAG_mode = 0;
int         FLAG_resolution     = 0;
std::string FLAG_codec          = DEFAULT_CODEC,
            FLAG_camRes         = "1280x720",
            FLAG_inFile,
            FLAG_outFile,
            FLAG_outDir,
            FLAG_modelDir,
            FLAG_effect;

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
    "VideoEffectsApp [args ...]\n"
    "  where args is:\n"
    "  --in_file=<path>           input file to be processed\n"
    "  --webcam                   use a webcam as the input\n"
    "  --out_file=<path>          output file to be written\n"
    "  --effect=<effect>          the effect to apply\n"
    "  --show                     display the results in a window (for webcam, it is always true)\n"
    "  --strength=<value>         strength of the upscaling effect, [0.0, 1.0]\n"
    "  --mode=<value>             mode of the super res or artifact reduction effect, 0 or 1, \n"
    "                             where 0 - conservative and 1 - aggressive\n"
    "  --cam_res=[WWWx]HHH        specify camera resolution as height or width x height\n"
    "                             supports 720 and 1080 resolutions (default \"720\") \n"
    "  --resolution=<height>      the desired height of the output\n"
    "  --model_dir=<path>         the path to the directory that contains the models\n"
    "  --codec=<fourcc>           the fourcc code for the desired codec (default " DEFAULT_CODEC ")\n"
    "  --progress                 show progress\n"
    "  --verbose                  verbose output\n"
    "  --debug                    print extra debugging information\n"
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
    if (arg[0] != '-') {
      continue;
    } else if ((arg[1] == '-') &&
      ( GetFlagArgVal("verbose",      arg, &FLAG_verbose)     ||
        GetFlagArgVal("in",           arg, &FLAG_inFile)      ||
        GetFlagArgVal("in_file",      arg, &FLAG_inFile)      ||
        GetFlagArgVal("out",          arg, &FLAG_outFile)     ||
        GetFlagArgVal("out_file",     arg, &FLAG_outFile)     ||
        GetFlagArgVal("effect",       arg, &FLAG_effect)      ||
        GetFlagArgVal("show",         arg, &FLAG_show)        ||
        GetFlagArgVal("webcam",       arg, &FLAG_webcam)      ||
        GetFlagArgVal("cam_res",      arg, &FLAG_camRes)      ||
        GetFlagArgVal("strength",     arg, &FLAG_strength)    ||
        GetFlagArgVal("mode",         arg, &FLAG_mode)        ||
        GetFlagArgVal("resolution",   arg, &FLAG_resolution)  ||
        GetFlagArgVal("model_dir",    arg, &FLAG_modelDir)    ||
        GetFlagArgVal("codec",        arg, &FLAG_codec)       ||
        GetFlagArgVal("progress",     arg, &FLAG_progress)    ||
        GetFlagArgVal("debug",        arg, &FLAG_debug)
        )) {
      continue;
    } else if (GetFlagArgVal("help", arg, &help)) {
      return NVCV_ERR_HELP;
    } else if (arg[1] != '-') {
      for (++arg; *arg; ++arg) {
        if (*arg == 'v') {
          FLAG_verbose = true;
        } else {
          printf("Unknown flag ignored: \"-%c\"\n", *arg);
        }
      }
      continue;
    } else {
      printf("Unknown flag ignored: \"%s\"\n", arg);
    }
  }
  return errs;
}

static bool HasSuffix(const char *str, const char *suf) {
  size_t  strSize = strlen(str),
    sufSize = strlen(suf);
  if (strSize < sufSize)
    return false;
  return (0 == strcasecmp(suf, str + strSize - sufSize));
}

static bool HasOneOfTheseSuffixes(const char *str, ...) {
  bool matches = false;
  const char *suf;
  va_list ap;
  va_start(ap, str);
  while (nullptr != (suf = va_arg(ap, const char*))) {
    if (HasSuffix(str, suf)) {
      matches = true;
      break;
    }
  }
  va_end(ap);
  return matches;
}

static bool IsImageFile(const char *str) {
  return HasOneOfTheseSuffixes(str, ".bmp", ".jpg", ".jpeg", ".png", nullptr);
}

static bool IsLossyImageFile(const char *str) {
  return HasOneOfTheseSuffixes(str, ".jpg", ".jpeg", nullptr);
}


static const char* DurationString(double sc) {
  static char buf[16];
  int         hr, mn;
  hr = (int)(sc / 3600.);
  sc -= hr * 3600.;
  mn = (int)(sc / 60.);
  sc -= mn * 60.;
  snprintf(buf, sizeof(buf), "%02d:%02d:%06.3f", hr, mn, sc);
  return buf;
}

struct VideoInfo {
  int         codec;
  int         width;
  int         height;
  double      frameRate;
  long long   frameCount;
};

static void GetVideoInfo(cv::VideoCapture& reader, const char *fileName, VideoInfo *info) {
  info->codec      =       (int)reader.get(cv::CAP_PROP_FOURCC);
  info->width      =       (int)reader.get(cv::CAP_PROP_FRAME_WIDTH);
  info->height     =       (int)reader.get(cv::CAP_PROP_FRAME_HEIGHT);
  info->frameRate  =    (double)reader.get(cv::CAP_PROP_FPS);
  info->frameCount = (long long)reader.get(cv::CAP_PROP_FRAME_COUNT);
  if (FLAG_verbose)
    printf(
      "       file \"%s\"\n"
      "      codec %.4s\n"
      "      width %4d\n"
      "     height %4d\n"
      " frame rate %.3f\n"
      "frame count %4lld\n"
      "   duration %s\n",
      fileName, (char*)&info->codec, info->width, info->height, info->frameRate, info->frameCount,
      DurationString(info->frameCount / info->frameRate)
    );
}

static int StringToFourcc(const std::string& str) {
    union chint { int i; char c[4]; };
    chint x = { 0 };
    for (int n = (str.size() < 4) ? (int)str.size() : 4; n--;)
      x.c[n] = str[n];
    return x.i;
}

struct FXApp {
  enum Err {
    errQuit               = +1,                         // Application errors
    errFlag               = +2,
    errRead               = +3,
    errWrite              = +4,
    errNone               = NVCV_SUCCESS,               // Video Effects SDK errors
    errGeneral            = NVCV_ERR_GENERAL,
    errUnimplemented      = NVCV_ERR_UNIMPLEMENTED,
    errMemory             = NVCV_ERR_MEMORY,
    errEffect             = NVCV_ERR_EFFECT,
    errSelector           = NVCV_ERR_SELECTOR,
    errBuffer             = NVCV_ERR_BUFFER,
    errParameter          = NVCV_ERR_PARAMETER,
    errMismatch           = NVCV_ERR_MISMATCH,
    errPixelFormat        = NVCV_ERR_PIXELFORMAT,
    errModel              = NVCV_ERR_MODEL,
    errLibrary            = NVCV_ERR_LIBRARY,
    errInitialization     = NVCV_ERR_INITIALIZATION,
    errFileNotFound       = NVCV_ERR_FILE,
    errFeatureNotFound    = NVCV_ERR_FEATURENOTFOUND,
    errMissingInput       = NVCV_ERR_MISSINGINPUT,
    errResolution         = NVCV_ERR_RESOLUTION,
    errUnsupportedGPU     = NVCV_ERR_UNSUPPORTEDGPU,
    errWrongGPU           = NVCV_ERR_WRONGGPU,
    errUnsupportedDriver  = NVCV_ERR_UNSUPPORTEDDRIVER,
    errCudaMemory         = NVCV_ERR_CUDA_MEMORY,       // CUDA errors
    errCudaValue          = NVCV_ERR_CUDA_VALUE,
    errCudaPitch          = NVCV_ERR_CUDA_PITCH,
    errCudaInit           = NVCV_ERR_CUDA_INIT,
    errCudaLaunch         = NVCV_ERR_CUDA_LAUNCH,
    errCudaKernel         = NVCV_ERR_CUDA_KERNEL,
    errCudaDriver         = NVCV_ERR_CUDA_DRIVER,
    errCudaUnsupported    = NVCV_ERR_CUDA_UNSUPPORTED,
    errCudaIllegalAddress = NVCV_ERR_CUDA_ILLEGAL_ADDRESS,
    errCuda               = NVCV_ERR_CUDA,
  };

  FXApp()   { _eff = nullptr; _effectName = nullptr; _inited = false; _showFPS = false; _progress = false;
              _show = false; _enableEffect = true, _drawVisualization = true, _framePeriod = 0.f; }
  ~FXApp()  { NvVFX_DestroyEffect(_eff); }

  void          setShow(bool show) { _show = show; }
  Err           createEffect(const char *effectSelector, const char *modelDir);
  void          destroyEffect();
  NvCV_Status   allocBuffers(unsigned width, unsigned height);
  NvCV_Status   allocTempBuffers();
  Err           processImage(const char *inFile, const char *outFile);
  Err           processMovie(const char *inFile, const char *outFile);
  Err           initCamera(cv::VideoCapture& cap);
  Err           processKey(int key);
  void          drawFrameRate(cv::Mat& img);
  void          drawEffectStatus(cv::Mat& img);
  Err           appErrFromVfxStatus(NvCV_Status status)  { return (Err)status; }
  const char*   errorStringFromCode(Err code);

  NvVFX_Handle  _eff;
  cv::Mat       _srcImg;
  cv::Mat       _dstImg;
  NvCVImage     _srcGpuBuf;
  NvCVImage     _dstGpuBuf;
  NvCVImage     _srcVFX;
  NvCVImage     _dstVFX;
  NvCVImage     _tmpVFX;  // We use the same temporary buffer for source and dst, since it auto-shapes as needed
  bool          _show;
  bool          _inited;
  bool          _showFPS;
  bool          _progress;
  bool          _enableEffect;
  bool          _drawVisualization;
  const char*   _effectName;
  float         _framePeriod;
  std::chrono::high_resolution_clock::time_point _lastTime;
};

const char* FXApp::errorStringFromCode(Err code) {
  struct LutEntry { Err code; const char *str; };
  static const LutEntry lut[] = {
    { errRead,    "There was a problem reading a file"                    },
    { errWrite,   "There was a problem writing a file"                    },
    { errQuit,    "The user chose to quit the application"                },
    { errFlag,    "There was a problem with the command-line arguments"   },
  };
  if ((int)code <= 0) return NvCV_GetErrorStringFromCode((NvCV_Status)code);
  for (const LutEntry *p = lut; p != &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
    if (p->code == code) return p->str;
  return "UNKNOWN ERROR";
}

void FXApp::drawFrameRate(cv::Mat &img) {
  const float timeConstant = 16.f;
  std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> dur = std::chrono::duration_cast<std::chrono::duration<float>>(now - _lastTime);
  float t = dur.count();
  if (0.f < t && t < 100.f) {
    if (_framePeriod)
      _framePeriod += (t - _framePeriod) * (1.f / timeConstant);  // 1 pole IIR filter
    else
      _framePeriod = t;
    if (_showFPS) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.1f", 1. / _framePeriod);
      cv::putText(img, buf, cv::Point(10, img.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
    }
  } else {            // Ludicrous time interval; reset
    _framePeriod = 0.f;  // WAKE UP
  }
  _lastTime = now;
}

FXApp::Err FXApp::processKey(int key) {
  static const int ESC_KEY = 27;
  switch (key) {
  case 'Q': case 'q': case ESC_KEY:
    return errQuit;
  case 'f': case 'F':
    _showFPS = !_showFPS;
    break;
  case 'p': case 'P': case '%':
    _progress = !_progress;
  case 'e': case 'E':
    break;
  case 'd': case'D':
    if (FLAG_webcam)
      _drawVisualization = !_drawVisualization;
    break;
  default:
    break;
  }
  return errNone;
}

FXApp::Err FXApp::initCamera(cv::VideoCapture& cap) {
  const int camIndex = 0;
  cap.open(camIndex);
  if (!FLAG_camRes.empty()) {
    int camWidth, camHeight, n;
    n = sscanf(FLAG_camRes.c_str(), "%d%*[xX]%d", &camWidth, &camHeight);
    switch (n) {
    case 2:
      break;  // We have read both width and height
    case 1:
      camHeight = camWidth;
      camWidth = (int)(camHeight * (16. / 9.) + .5);
      break;
    default:
      camHeight = 0;
      camWidth = 0;
      break;
    }

    if (camWidth) cap.set(cv::CAP_PROP_FRAME_WIDTH, camWidth);
    if (camHeight) cap.set(cv::CAP_PROP_FRAME_HEIGHT, camHeight);
    if (camWidth != cap.get(cv::CAP_PROP_FRAME_WIDTH) || camHeight != cap.get(cv::CAP_PROP_FRAME_HEIGHT)) {
      printf("Error: Camera does not support %d x %d resolution\n", camWidth, camHeight);
      return errGeneral;
    }
  }
  return errNone;
}

void FXApp::drawEffectStatus(cv::Mat& img) {
  char buf[32];
  snprintf(buf, sizeof(buf), "Effect: %s", _enableEffect ? "on" : "off");
  cv::putText(img, buf, cv::Point(10, img.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
}

FXApp::Err FXApp::createEffect(const char *effectSelector, const char *modelDir) {
  NvCV_Status vfxErr;
  BAIL_IF_ERR(vfxErr = NvVFX_CreateEffect(effectSelector, &_eff));
  _effectName = effectSelector;
  if (modelDir[0] != '\0'){
    BAIL_IF_ERR(vfxErr = NvVFX_SetString(_eff, NVVFX_MODEL_DIRECTORY, modelDir));
  }
bail:
  return appErrFromVfxStatus(vfxErr);
}

void FXApp::destroyEffect() {
  NvVFX_DestroyEffect(_eff);
  _eff = nullptr;
}

// Allocate one temp buffer to be used for input and output. Reshaping of the temp buffer in NvCVImage_Transfer() is done automatically,
// and is very low overhead. We expect the destination to be largest, so we allocate that first to minimize reallocs probablistically.
// Then we Realloc for the source to get the union of the two.
// This could alternately be done at runtime by feeding in an empty temp NvCVImage, but there are advantages to allocating all memory at load time.
NvCV_Status FXApp::allocTempBuffers() {
  NvCV_Status vfxErr;
  BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(  &_tmpVFX, _dstVFX.width, _dstVFX.height, _dstVFX.pixelFormat, _dstVFX.componentType, _dstVFX.planar, NVCV_GPU, 0));
  BAIL_IF_ERR(vfxErr = NvCVImage_Realloc(&_tmpVFX, _srcVFX.width, _srcVFX.height, _srcVFX.pixelFormat, _srcVFX.componentType, _srcVFX.planar, NVCV_GPU, 0));
bail:
  return vfxErr;
}

static NvCV_Status CheckScaleIsotropy(const NvCVImage *src, const NvCVImage *dst) {
  if (src->width * dst->height != src->height * dst->width) {
    printf("%ux%u --> %ux%u: different scale for width and height is not supported\n",
      src->width, src->height, dst->width, dst->height);
    return NVCV_ERR_RESOLUTION;
  }
  return NVCV_SUCCESS;
}

NvCV_Status FXApp::allocBuffers(unsigned width, unsigned height) {
  NvCV_Status  vfxErr = NVCV_SUCCESS;

  if (_inited)
    return NVCV_SUCCESS;

  if (!_srcImg.data) {
    _srcImg.create(height, width, CV_8UC3);                                                                                        // src CPU
    BAIL_IF_NULL(_srcImg.data, vfxErr, NVCV_ERR_MEMORY);
  }
  if (!strcmp(_effectName, NVVFX_FX_TRANSFER)) {
    _dstImg.create(_srcImg.rows, _srcImg.cols, _srcImg.type());                                                                    // dst CPU
    BAIL_IF_NULL(_dstImg.data, vfxErr, NVCV_ERR_MEMORY);
    BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcGpuBuf, _srcImg.cols, _srcImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // src GPU
    BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstGpuBuf, _dstImg.cols, _dstImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // dst GPU
  }
  else if (!strcmp(_effectName, NVVFX_FX_ARTIFACT_REDUCTION)) {
    _dstImg.create(_srcImg.rows, _srcImg.cols, _srcImg.type());                                                                    // dst CPU
    BAIL_IF_NULL(_dstImg.data, vfxErr, NVCV_ERR_MEMORY);
    BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcGpuBuf, _srcImg.cols, _srcImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // src GPU
    BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstGpuBuf, _dstImg.cols, _dstImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // dst GPU
  }
  else if (!strcmp(_effectName, NVVFX_FX_SUPER_RES)) {
    if (!FLAG_resolution) {
      printf("--resolution has not been specified\n");
      return NVCV_ERR_PARAMETER;
    }
    int dstWidth = _srcImg.cols * FLAG_resolution / _srcImg.rows;
    _dstImg.create(FLAG_resolution, dstWidth, _srcImg.type());                                                                     // dst CPU
    BAIL_IF_NULL(_dstImg.data, vfxErr, NVCV_ERR_MEMORY);
    BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcGpuBuf, _srcImg.cols, _srcImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // src GPU
    BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstGpuBuf, _dstImg.cols, _dstImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // dst GPU
    BAIL_IF_ERR(vfxErr = CheckScaleIsotropy(&_srcGpuBuf, &_dstGpuBuf));
  }
  else if (!strcmp(_effectName, NVVFX_FX_SR_UPSCALE)) {
    if (!FLAG_resolution) {
      printf("--resolution has not been specified\n");
      return NVCV_ERR_PARAMETER;
    }

    BAIL_IF_ERR(vfxErr = NvVFX_SetF32(_eff, NVVFX_STRENGTH, FLAG_strength));
    int dstWidth = _srcImg.cols * FLAG_resolution / _srcImg.rows;
    _dstImg.create(FLAG_resolution, dstWidth, _srcImg.type());  // dst CPU
    BAIL_IF_NULL(_dstImg.data, vfxErr, NVCV_ERR_MEMORY);
    BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcGpuBuf, _srcImg.cols, _srcImg.rows, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED,
                                         NVCV_GPU, 32));  // src GPU
    BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstGpuBuf, _dstImg.cols, _dstImg.rows, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED,
                                         NVCV_GPU, 32));  // dst GPU
    BAIL_IF_ERR(vfxErr = CheckScaleIsotropy(&_srcGpuBuf, &_dstGpuBuf));
  }
  NVWrapperForCVMat(&_srcImg, &_srcVFX);      // _srcVFX is an alias for _srcImg
  NVWrapperForCVMat(&_dstImg, &_dstVFX);      // _dstVFX is an alias for _dstImg

  //#define ALLOC_TEMP_BUFFERS_AT_RUN_TIME    // Deferring temp buffer allocation is easier
  #ifndef ALLOC_TEMP_BUFFERS_AT_RUN_TIME      // Allocating temp buffers at load time avoids run time hiccups
    BAIL_IF_ERR(vfxErr = allocTempBuffers()); // This uses _srcVFX and _dstVFX and allocates one buffer to be a temporary for src and dst
  #endif // ALLOC_TEMP_BUFFERS_AT_RUN_TIME

  _inited = true;

bail:
  return vfxErr;
}

FXApp::Err FXApp::processImage(const char *inFile, const char *outFile) {
  CUstream      stream  = 0;
  NvCV_Status   vfxErr;

  if (!_eff)
    return errEffect;
  _srcImg = cv::imread(inFile);
  if (!_srcImg.data)
    return errRead;

  BAIL_IF_ERR(vfxErr = allocBuffers(_srcImg.cols, _srcImg.rows));

  // Since images are uploaded asynchronously, we may as well do this first.
  BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_srcVFX, &_srcGpuBuf, 1.f/255.f, stream, &_tmpVFX)); // _srcVFX--> _tmpVFX --> _srcGpuBuf
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_INPUT_IMAGE,  &_srcGpuBuf));
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_OUTPUT_IMAGE, &_dstGpuBuf));
  BAIL_IF_ERR(vfxErr = NvVFX_SetCudaStream(_eff, NVVFX_CUDA_STREAM, stream));
  if (!strcmp(_effectName, NVVFX_FX_ARTIFACT_REDUCTION)) {
    BAIL_IF_ERR(vfxErr = NvVFX_SetU32(_eff, NVVFX_MODE, (unsigned int)FLAG_mode));
  } else if (!strcmp(_effectName, NVVFX_FX_SUPER_RES)) {
    BAIL_IF_ERR(vfxErr = NvVFX_SetU32(_eff, NVVFX_MODE, (unsigned int)FLAG_mode));
  }

  BAIL_IF_ERR(vfxErr = NvVFX_Load(_eff));
  BAIL_IF_ERR(vfxErr = NvVFX_Run(_eff, 0));                                                   // _srcGpuBuf --> _dstGpuBuf
  BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_dstGpuBuf, &_dstVFX, 255.f, stream, &_tmpVFX));   // _dstGpuBuf --> _tmpVFX --> _dstVFX

  if (outFile && outFile[0]) {
    if(IsLossyImageFile(outFile))
      fprintf(stderr, "WARNING: JPEG output file format will reduce image quality\n");
    if (!cv::imwrite(outFile, _dstImg)) {
      printf("Error writing: \"%s\"\n", outFile);
      return errWrite;
    }
  }
  if (_show) {
    cv::imshow("Output", _dstImg);
    cv::waitKey(3000);
  }
bail:
  return appErrFromVfxStatus(vfxErr);
}

FXApp::Err FXApp::processMovie(const char *inFile, const char *outFile) {
  const int       fourcc_h264 = cv::VideoWriter::fourcc('H','2','6','4');
  CUstream        stream      = 0;
  FXApp::Err      appErr      = errNone;
  bool            ok;
  cv::VideoCapture reader;
  cv::VideoWriter writer;
  NvCV_Status     vfxErr;
  unsigned        frameNum;
  VideoInfo       info;

  if (inFile && !inFile[0]) inFile = nullptr;  // Set file paths to NULL if zero length

  if (!FLAG_webcam && inFile) {
    reader.open(inFile);
  } else {
    appErr = initCamera(reader);
    if (appErr != errNone)
      return appErr;
  }

  if (!reader.isOpened()) {
    if (!FLAG_webcam) printf("Error: Could not open video: \"%s\"\n", inFile);
    else              printf("Error: Webcam not found\n");
    return errRead;
  }

  GetVideoInfo(reader, (inFile ? inFile : "webcam"), &info);
  if (!(fourcc_h264 == info.codec || cv::VideoWriter::fourcc('a', 'v', 'c', '1') == info.codec)) // avc1 is alias for h264
    printf("Filters only target H264 videos, not %.4s\n", (char*)&info.codec);

  BAIL_IF_ERR(vfxErr = allocBuffers(info.width, info.height));

  if (outFile && !outFile[0]) outFile = nullptr;
  if (outFile) {
    ok = writer.open(outFile, StringToFourcc(FLAG_codec), info.frameRate, cv::Size(_dstVFX.width, _dstVFX.height));
    if (!ok) {
      printf("Cannot open \"%s\" for video writing\n", outFile);
      outFile = nullptr;
      if (!_show)
        return errWrite;
    }
  }

  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_INPUT_IMAGE,  &_srcGpuBuf));
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_OUTPUT_IMAGE, &_dstGpuBuf));
  BAIL_IF_ERR(vfxErr = NvVFX_SetCudaStream(_eff, NVVFX_CUDA_STREAM, stream));
  if (!strcmp(_effectName, NVVFX_FX_ARTIFACT_REDUCTION)) {
    BAIL_IF_ERR(vfxErr = NvVFX_SetU32(_eff, NVVFX_MODE, (unsigned int)FLAG_mode));
  } else if (!strcmp(_effectName, NVVFX_FX_SUPER_RES)) {
    BAIL_IF_ERR(vfxErr = NvVFX_SetU32(_eff, NVVFX_MODE, (unsigned int)FLAG_mode));
  }
  BAIL_IF_ERR(vfxErr = NvVFX_Load(_eff));

  for (frameNum = 0; reader.read(_srcImg); ++frameNum) {
    if (_srcImg.empty()) {
      printf("Frame %u is empty\n", frameNum);
    }

    // _srcVFX   --> _srcTmpVFX --> _srcGpuBuf --> _dstGpuBuf --> _dstTmpVFX --> _dstVFX
    if (_enableEffect) {
      BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_srcVFX, &_srcGpuBuf, 1.f / 255.f, stream, &_tmpVFX));
      BAIL_IF_ERR(vfxErr = NvVFX_Run(_eff, 0));
      BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_dstGpuBuf, &_dstVFX, 255.f, stream, &_tmpVFX));
    } else {
      BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_srcVFX, &_dstVFX, 1.f / 255.f, stream, &_tmpVFX));
    }

    if (outFile)
      writer.write(_dstImg);

    if (_show) {
      drawFrameRate(_dstImg);
      cv::imshow("Output", _dstImg);
      int key= cv::waitKey(1);
      if (key > 0) {
          appErr = processKey(key);
          if (errQuit == appErr)
            break;
      }
    }
    if (_progress)
      fprintf(stderr, "\b\b\b\b%3.0f%%", 100.f * frameNum / info.frameCount);
  }

  if (_progress) fprintf(stderr, "\n");
  reader.release();
  if (outFile)
    writer.release();
bail:
  return appErrFromVfxStatus(vfxErr);
}

int main(int argc, char **argv) {
  FXApp::Err  fxErr = FXApp::errNone;
  int         nErrs;
  FXApp       app;

  nErrs = ParseMyArgs(argc, argv);
  if (nErrs)
    std::cerr << nErrs << " command line syntax problems\n";

  if (FLAG_verbose) {
    const char *cstr = nullptr;
    NvVFX_GetString(nullptr, NVVFX_INFO, &cstr);
    std::cerr << "Effects:" << std::endl << cstr << std::endl;
  }
  if (FLAG_webcam) {
    // If webcam is on, enable showing the results and turn off displaying the progress
    if (FLAG_progress) FLAG_progress = !FLAG_progress;
    if (!FLAG_show)     FLAG_show = !FLAG_show;
  }
  if (FLAG_inFile.empty() && !FLAG_webcam) {
    std::cerr << "Please specify --in_file=XXX or --webcam=true\n";
    ++nErrs;
  }
  if (FLAG_outFile.empty() && !FLAG_show) {
    std::cerr << "Please specify --out_file=XXX or --show\n";
    ++nErrs;
  }
  if (FLAG_effect.empty()) {
    std::cerr << "Please specify --effect=XXX\n";
    ++nErrs;
  }
  app._progress = FLAG_progress;
  app.setShow(FLAG_show);

  if (nErrs) {
    Usage();
    fxErr = FXApp::errFlag;
  }
  else {
    fxErr = app.createEffect(FLAG_effect.c_str(), FLAG_modelDir.c_str());
    if (FXApp::errNone != fxErr) {
      std::cerr << "Error creating effect \"" << FLAG_effect << "\"\n";
    }
    else {
      if (IsImageFile(FLAG_inFile.c_str()))
        fxErr = app.processImage(FLAG_inFile.c_str(), FLAG_outFile.c_str());
      else
        fxErr = app.processMovie(FLAG_inFile.c_str(), FLAG_outFile.c_str());
    }
  }

  if (fxErr)
    std::cerr << "Error: " << app.errorStringFromCode(fxErr) << std::endl;
  return (int)fxErr;
}

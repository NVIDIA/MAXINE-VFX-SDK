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

#include "nvCVOpenCV.h"
#include "nvVideoEffects.h"
#include "opencv2/opencv.hpp"

/*########################################################################################################################
# This application demonstrates the pipelining of two NvVFX_API video effects through a common use case whereby an image
# or image sequence is fed first through the Artifact Removal filter, and then through the Super Resolution filter,
# to produce an upscaled, video compression artifact-reduced version of the image/image sequence.
# This is likely to be useful when dealing with low-quality input video bitstreams,
# such as during game or movie streaming in a congested network environment.
# While only the specific use case of pipelining the Artifact Removal and Super Resolution
# filters is supported here to avoid undue code complexity, the basic method and structure shown here can be applied
# to pipeline an arbitrary sequence of NvVFX_API video effects.
##########################################################################################################################*/

#ifdef _MSC_VER
  #define strcasecmp _stricmp
  #include <Windows.h>
#else // !_MSC_VER
  #include <sys/stat.h>
#endif // _MSC_VER

#define BAIL_IF_ERR(err)            do { if (0 != (err))          { goto bail;             } } while(0)
#define BAIL_IF_NULL(x, err, code)  do { if ((void*)(x) == NULL)  { err = code; goto bail; } } while(0)


bool        FLAG_debug               = false,
            FLAG_verbose             = false,
            FLAG_show                = false,
            FLAG_progress            = false;
int         FLAG_resolution          = 0,
            FLAG_arStrength          = 0;
float       FLAG_upscaleStrength     = 0.f;
std::string FLAG_codec               = "H264",
            FLAG_inFile,
            FLAG_outFile,
            FLAG_outDir,
            FLAG_modelDir;

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
    "UpscalePipelineApp [args ...]\n"
    "  where args is:\n"
    "  --in_file=<path>                    input file to be processed\n"
    "  --out_file=<path>                   output file to be written\n"
    "  --show                              display the results in a window\n"
    "  --ar_strength=(0|1)                 strength of artifact reduction filter (0: conservative, 1: aggressive, default 0)\n"
    "  --upscale_strength=(0 to 1)         strength of upscale filter (float value between 0 to 1)\n"
    "  --resolution=<height>               the desired height of the output\n"
    "  --out_height=<height>               the desired height of the output\n"
    "  --model_dir=<path>                  the path to the directory that contains the models\n"
    "  --codec=<fourcc>                    the fourcc code for the desired codec (default \"H264\")\n"
    "  --progress                          show progress\n"
    "  --verbose                           verbose output\n"
    "  --debug                             print extra debugging information\n"
  );
}

static int ParseMyArgs(int argc, char **argv) {
  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char *arg = *argv;
    if (arg[0] != '-') {
      continue;
    } else if ((arg[1] == '-') &&
      ( GetFlagArgVal("verbose",          arg, &FLAG_verbose)     ||
        GetFlagArgVal("in",               arg, &FLAG_inFile)      ||
        GetFlagArgVal("in_file",          arg, &FLAG_inFile)      ||
        GetFlagArgVal("out",              arg, &FLAG_outFile)     ||
        GetFlagArgVal("out_file",         arg, &FLAG_outFile)     ||
        GetFlagArgVal("show",             arg, &FLAG_show)        ||
        GetFlagArgVal("ar_strength",      arg, &FLAG_arStrength)  ||
        GetFlagArgVal("upscale_strength", arg, &FLAG_upscaleStrength)  ||
        GetFlagArgVal("resolution",       arg, &FLAG_resolution)  ||
        GetFlagArgVal("model_dir",        arg, &FLAG_modelDir)    ||
        GetFlagArgVal("codec",            arg, &FLAG_codec)       ||
        GetFlagArgVal("progress",         arg, &FLAG_progress)    ||
        GetFlagArgVal("debug",            arg, &FLAG_debug)
        )) {
      continue;
    } else if (GetFlagArgVal("help", arg, &help)) {
      Usage();
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
    errQuit               = -50,                        // Application errors
    errFlag               = -51,
    errRead               = -52,
    errWrite              = -53,
  };

  FXApp()   { _arEff = nullptr; _upscaleEff = nullptr; _inited = false; _showFPS = false; _progress = false;
              _show = false; _framePeriod = 0.f; }
  ~FXApp()  { destroyEffects(); }

  void          setShow(bool show) { _show = show; }
  Err           createEffects(const char *modelDir, NvVFX_EffectSelector first, NvVFX_EffectSelector second);
  void          destroyEffects();
  NvCV_Status   allocBuffers(unsigned width, unsigned height);
  NvCV_Status   allocTempBuffers();
  Err           processImage(const char *inFile, const char *outFile);
  Err           processMovie(const char *inFile, const char *outFile);
  Err           processKey(int key);
  void          drawFrameRate(cv::Mat& img);
  Err           appErrFromVfxStatus(NvCV_Status status)  { return (Err)status; }
  const char*   errorStringFromCode(Err code);

  NvVFX_Handle  _arEff;
  NvVFX_Handle  _upscaleEff;
  cv::Mat       _srcImg;
  cv::Mat       _dstImg;
  NvCVImage     _srcGpuBuf;
  NvCVImage     _interGpuBGRf32pl;
  NvCVImage     _interGpuRGBAu8;
  NvCVImage     _dstGpuBuf;
  NvCVImage     _srcVFX;
  NvCVImage     _dstVFX;
  NvCVImage     _tmpVFX;
  bool          _show;
  bool          _inited;
  bool          _showFPS;
  bool          _progress;
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
  if ((int)code >= (int)errCuda)
    return NvCV_GetErrorStringFromCode((NvCV_Status)code);
  for (const LutEntry *p = lut; p != &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
    if (p->code == code)
      return p->str;
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
    break;
  default:
    break;
  }
  return errNone;
}

FXApp::Err FXApp::createEffects(const char *modelDir, NvVFX_EffectSelector first, NvVFX_EffectSelector second) {
  NvCV_Status vfxErr;
  BAIL_IF_ERR(vfxErr = NvVFX_CreateEffect(first, &_arEff));
  BAIL_IF_ERR(vfxErr = NvVFX_SetString(_arEff, NVVFX_MODEL_DIRECTORY, modelDir));
  BAIL_IF_ERR(vfxErr = NvVFX_CreateEffect(second, &_upscaleEff));
  if (modelDir[0] != '\0'){
    BAIL_IF_ERR(vfxErr = NvVFX_SetString(_upscaleEff, NVVFX_MODEL_DIRECTORY, modelDir));
  }
bail:
  return appErrFromVfxStatus(vfxErr);
}

void FXApp::destroyEffects() {
  NvVFX_DestroyEffect(_arEff);
  _arEff = nullptr;
  NvVFX_DestroyEffect(_upscaleEff);
  _upscaleEff = nullptr;
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

NvCV_Status FXApp::allocBuffers(unsigned width, unsigned height) {
  NvCV_Status vfxErr = NVCV_SUCCESS;
  int dstWidth;

  if (_inited)
    return NVCV_SUCCESS;

  if (!_srcImg.data) {
    _srcImg.create(height, width, CV_8UC3);                                                         // src CPU
    BAIL_IF_NULL(_srcImg.data, vfxErr, NVCV_ERR_MEMORY);
  }

  if (!FLAG_resolution) {
    printf("--resolution has not been specified\n");
    return NVCV_ERR_PARAMETER;
  }
  dstWidth = _srcImg.cols * FLAG_resolution / _srcImg.rows;
  _dstImg.create(FLAG_resolution, dstWidth, _srcImg.type());                                        // dst CPU
  BAIL_IF_NULL(_dstImg.data, vfxErr, NVCV_ERR_MEMORY);
  BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcGpuBuf,   _srcImg.cols, _srcImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // src GPU
  BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_interGpuBGRf32pl, _srcImg.cols, _srcImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // intermediate GPU
  BAIL_IF_ERR(vfxErr = NvVFX_SetF32(_upscaleEff, NVVFX_STRENGTH, FLAG_upscaleStrength));
  BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_interGpuRGBAu8, _srcImg.cols, _srcImg.rows, NVCV_RGBA, NVCV_U8,
                                       NVCV_INTERLEAVED, NVCV_GPU, 1));                             // intermediate GPU

  BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstGpuBuf, _dstImg.cols, _dstImg.rows, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED,
                                       NVCV_GPU, 1));                                               // dst GPU
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
  CUstream    stream  = 0;
  NvCV_Status vfxErr;

  if (!_arEff || !_upscaleEff)
    return errEffect;
  _srcImg = cv::imread(inFile);
  if (!_srcImg.data)
    return errRead;

  BAIL_IF_ERR(vfxErr = allocBuffers(_srcImg.cols, _srcImg.rows));

  BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_srcVFX, &_srcGpuBuf, 1.f/255.f, stream, &_tmpVFX)); // _srcTmpVFX--> _dstTmpVFX --> _srcGpuBuf
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_arEff, NVVFX_INPUT_IMAGE,  &_srcGpuBuf));
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_arEff, NVVFX_OUTPUT_IMAGE, &_interGpuBGRf32pl));
  BAIL_IF_ERR(vfxErr = NvVFX_SetCudaStream(_arEff, NVVFX_CUDA_STREAM, stream));
  BAIL_IF_ERR(vfxErr = NvVFX_SetU32(_arEff, NVVFX_STRENGTH, FLAG_arStrength));

  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_upscaleEff, NVVFX_INPUT_IMAGE, &_interGpuRGBAu8));
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_upscaleEff, NVVFX_OUTPUT_IMAGE, &_dstGpuBuf));
  BAIL_IF_ERR(vfxErr = NvVFX_SetCudaStream(_upscaleEff, NVVFX_CUDA_STREAM, stream));

  BAIL_IF_ERR(vfxErr = NvVFX_Load(_arEff));
  BAIL_IF_ERR(vfxErr = NvVFX_Load(_upscaleEff));
  BAIL_IF_ERR(vfxErr = NvVFX_Run(_arEff, 0));                                             // _srcGpuBuf --> _interGpuBuf
  // transfer between intermediate buffers if selected method is Upscale
  BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_interGpuBGRf32pl, &_interGpuRGBAu8, 255.f, stream, &_tmpVFX));
  BAIL_IF_ERR(vfxErr = NvVFX_Run(_upscaleEff, 0));                                        // _interGpuBuf --> _dstGpuBuf
  BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_dstGpuBuf, &_dstVFX, 1.f, stream, &_tmpVFX)); // _dstGpuBuf --> _dstTmpVFX --> _dstVFX

  if (outFile && outFile[0]) {
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
  const int       fourcc_h264 = CV_FOURCC('H','2','6','4');
  CUstream        stream      = 0;
  FXApp::Err      appErr      = errNone;
  bool            ok;
  cv::VideoWriter writer;
  NvCV_Status     vfxErr;
  unsigned        frameNum;
  VideoInfo       info;

  cv::VideoCapture reader(inFile);
  if (!reader.isOpened()) {
    printf("Error: Could not open video: \"%s\"\n", inFile);
    return errRead;
  }

  GetVideoInfo(reader, inFile, &info);
  if (!(fourcc_h264 == info.codec || CV_FOURCC('a','v','c','1') == info.codec)) // avc1 is alias for h264
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

  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_arEff, NVVFX_INPUT_IMAGE,  &_srcGpuBuf));
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_arEff, NVVFX_OUTPUT_IMAGE, &_interGpuBGRf32pl));
  BAIL_IF_ERR(vfxErr = NvVFX_SetCudaStream(_arEff, NVVFX_CUDA_STREAM, stream));
  BAIL_IF_ERR(vfxErr = NvVFX_SetU32(_arEff, NVVFX_STRENGTH, FLAG_arStrength));
  BAIL_IF_ERR(vfxErr = NvVFX_Load(_arEff));

  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_upscaleEff, NVVFX_INPUT_IMAGE, &_interGpuRGBAu8));
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_upscaleEff, NVVFX_OUTPUT_IMAGE, &_dstGpuBuf));
  BAIL_IF_ERR(vfxErr = NvVFX_SetCudaStream(_upscaleEff, NVVFX_CUDA_STREAM, stream));
  BAIL_IF_ERR(vfxErr = NvVFX_Load(_upscaleEff));

  for (frameNum = 0; reader.read(_srcImg); ++frameNum) {
    if (_srcImg.empty()) {
      printf("Frame %u is empty\n", frameNum);
    }

    // _srcVFX   --> _srcTmpVFX --> _srcGpuBuf --> _interGpuBuf --> _dstGpuBuf --> _dstTmpVFX --> _dstVFX
    BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_srcVFX, &_srcGpuBuf, 1.f/255.f, stream, &_tmpVFX));
    BAIL_IF_ERR(vfxErr = NvVFX_Run(_arEff, 0));
    // transfer between intermediate buffers if selected method is Upscale
    BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_interGpuBGRf32pl, &_interGpuRGBAu8, 255.f, stream, &_tmpVFX));
    BAIL_IF_ERR(vfxErr = NvVFX_Run(_upscaleEff, 0));
    BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_dstGpuBuf, &_dstVFX, 1.f, stream, &_tmpVFX));

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
  int         nErrs = 0;
  FXApp::Err  fxErr = FXApp::errNone;
  FXApp       app;

  nErrs = ParseMyArgs(argc, argv);
  if (nErrs)
    std::cerr << nErrs << " command line syntax problems\n";

  if (FLAG_inFile.empty()) {
    std::cerr << "Please specify --in_file=XXX\n";
    ++nErrs;
  }
  if (FLAG_outFile.empty() && !FLAG_show) {
    std::cerr << "Please specify --out_file=XXX or --show\n";
    ++nErrs;
  }

  app._progress = FLAG_progress;
  app.setShow(FLAG_show);

  if (nErrs) {
    Usage();
    fxErr = FXApp::errFlag;
  }
  else {
    NvVFX_EffectSelector first = NVVFX_FX_ARTIFACT_REDUCTION;
    NvVFX_EffectSelector second = NVVFX_FX_SR_UPSCALE;

    fxErr = app.createEffects(FLAG_modelDir.c_str(), first, second);
    if (FXApp::errNone != fxErr) {
      std::cerr << "Error creating effects \"" << first << " & " << second << "\"\n";
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

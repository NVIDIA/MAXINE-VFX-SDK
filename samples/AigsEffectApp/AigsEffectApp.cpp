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
#include <time.h>

#include <chrono>
#include <string>
#include <iostream>
#include <vector>

#include "nvCVOpenCV.h"
#include "nvVideoEffects.h"
#include "opencv2/opencv.hpp"

#ifdef _MSC_VER
#define strcasecmp _stricmp
#include <Windows.h>
#else  // !_MSC_VER
#include <sys/stat.h>
#endif  // _MSC_VER

#define BAIL_IF_ERR(err) \
  do {                   \
    if (0 != (err)) {    \
      goto bail;         \
    }                    \
  } while (0)
#define BAIL_IF_NULL(x, err, code) \
  do {                             \
    if ((void *)(x) == NULL) {     \
      err = code;                  \
      goto bail;                   \
    }                              \
  } while (0)
#define NVCV_ERR_HELP 411

#ifdef _WIN32
  #define DEFAULT_CODEC "avc1"
#else // !_WIN32
  #define DEFAULT_CODEC "H264"
#endif // _WIN32

bool        FLAG_progress = false;
bool        FLAG_show     = false;
bool        FLAG_verbose  = false;
bool        FLAG_webcam   = false;
bool        FLAG_cudaGraph = false;
int         FLAG_compMode = 3 /*compWhite*/;
int         FLAG_mode     = 0;
float       FLAG_blurStrength = 0.5;
std::string FLAG_camRes;
std::string FLAG_codec    = DEFAULT_CODEC;
std::string FLAG_inFile;
std::string FLAG_modelDir;
std::string FLAG_outDir;
std::string FLAG_outFile;
std::string FLAG_bgFile;

static bool GetFlagArgVal(const char *flag, const char *arg, const char **val) {
  if (*arg != '-') return false;
  while (*++arg == '-') continue;
  const char *s = strchr(arg, '=');
  if (s == NULL) {
    if (strcmp(flag, arg) != 0) return false;
    *val = NULL;
    return true;
  }
  size_t n = s - arg;
  if ((strlen(flag) != n) || (strncmp(flag, arg, n) != 0)) return false;
  *val = s + 1;
  return true;
}

static bool GetFlagArgVal(const char *flag, const char *arg, std::string *val) {
  const char *valStr;
  if (!GetFlagArgVal(flag, arg, &valStr)) return false;
  val->assign(valStr ? valStr : "");
  return true;
}

static bool GetFlagArgVal(const char *flag, const char *arg, bool *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) {
    *val = (valStr == NULL || strcasecmp(valStr, "true") == 0 || strcasecmp(valStr, "on") == 0 ||
            strcasecmp(valStr, "yes") == 0 || strcasecmp(valStr, "1") == 0);
  }
  return success;
}

static bool GetFlagArgVal(const char *flag, const char *arg, long *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) *val = strtol(valStr, NULL, 10);
  return success;
}

static bool GetFlagArgVal(const char *flag, const char *arg, int *val) {
  long longVal;
  bool success = GetFlagArgVal(flag, arg, &longVal);
  if (success) *val = (int)longVal;
  return success;
}

static bool GetFlagArgVal(const char *flag, const char *arg, float *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) *val = std::stof(valStr);
  return success;
}

static void Usage() {
  printf(
      "AigsEffectApp [args ...]\n"
      "  where args is:\n"
      "  --in_file=<path>           input file to be processed\n"
      "  --out_file=<path>          output file to be written\n"
      "  --bg_file=<path>           background file for composition\n"
      "  --webcam                   use a webcam as input\n"
      "  --cam_res=[WWWx]HHH        specify resolution as height or width x height\n"
      "  --model_dir=<path>         the path to the directory that contains the models\n"
      "  --codec=<fourcc>           the FOURCC code for the desired codec (default " DEFAULT_CODEC ")\n"
      "  --show                     display the results in a window\n"
      "  --progress                 show progress\n"
      "  --mode=(0|1)               pick one of the green screen modes\n"
      "                             0 - Best quality\n"
      "                             1 - Best performance\n"
      "  --comp_mode                choose the composition mode - {\n"
      "                               0 (show matte - compMatte),\n"
      "                               1 (overlay mask on foreground - compLight),\n"
      "                               2 (composite over green - compGreen),\n"
      "                               3 (composite over white - compWhite),\n"
      "                               4 (show input - compNone),\n"
      "                               5 (composite over a specified background image - compBG),\n"
      "                               6 (blur the background of the image - compBlur) }\n"
      "  --blur_strength=[0-1]      strength of the background blur, when applicable\n"
      "  --cuda_graph               Enable cuda graph.\n"
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
               (GetFlagArgVal("verbose", arg, &FLAG_verbose) || GetFlagArgVal("in", arg, &FLAG_inFile) ||
                GetFlagArgVal("in_file", arg, &FLAG_inFile) || GetFlagArgVal("out", arg, &FLAG_outFile) ||
                GetFlagArgVal("out_file", arg, &FLAG_outFile) || GetFlagArgVal("model_dir", arg, &FLAG_modelDir) ||
                GetFlagArgVal("bg_file", arg, &FLAG_bgFile) ||
                GetFlagArgVal("codec", arg, &FLAG_codec) || GetFlagArgVal("webcam", arg, &FLAG_webcam) ||
                GetFlagArgVal("cam_res", arg, &FLAG_camRes) || GetFlagArgVal("mode", arg, &FLAG_mode) ||
                GetFlagArgVal("progress", arg, &FLAG_progress) || GetFlagArgVal("show", arg, &FLAG_show) ||
                GetFlagArgVal("comp_mode", arg, &FLAG_compMode) || GetFlagArgVal("blur_strength", arg, &FLAG_blurStrength) ||
                GetFlagArgVal("cuda_graph", arg, &FLAG_cudaGraph) )) {
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
  size_t strSize = strlen(str), sufSize = strlen(suf);
  if (strSize < sufSize) return false;
  return (0 == strcasecmp(suf, str + strSize - sufSize));
}

static bool HasOneOfTheseSuffixes(const char *str, ...) {
  bool matches = false;
  const char *suf;
  va_list ap;
  va_start(ap, str);
  while (nullptr != (suf = va_arg(ap, const char *))) {
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

static const char *DurationString(double sc) {
  static char buf[16];
  int hr, mn;
  hr = (int)(sc / 3600.);
  sc -= hr * 3600.;
  mn = (int)(sc / 60.);
  sc -= mn * 60.;
  snprintf(buf, sizeof(buf), "%02d:%02d:%06.3f", hr, mn, sc);
  return buf;
}

struct VideoInfo {
  int codec;
  int width;
  int height;
  double frameRate;
  long long frameCount;
};

static void PrintVideoInfo(const VideoInfo *info, const char *fileName) {
  printf(
      "       file \"%s\"\n"
      "      codec %.4s\n"
      "      width %4d\n"
      "     height %4d\n"
      " frame rate %.3f\n"
      "frame count %4lld\n"
      "   duration %s\n",
      fileName, (char *)&info->codec, info->width, info->height, info->frameRate, info->frameCount,
      info->frameCount ? DurationString(info->frameCount / info->frameRate) : "(webcam)");
}

static void GetVideoInfo(cv::VideoCapture &reader, const char *fileName, VideoInfo *info) {
  info->codec = (int)reader.get(cv::CAP_PROP_FOURCC);
  info->width = (int)reader.get(cv::CAP_PROP_FRAME_WIDTH);
  info->height = (int)reader.get(cv::CAP_PROP_FRAME_HEIGHT);
  info->frameRate = (double)reader.get(cv::CAP_PROP_FPS);
  if(!strcmp(fileName,"webcam"))
      info->frameCount = 0;
  else
      info->frameCount = (long long)reader.get(cv::CAP_PROP_FRAME_COUNT);
  if (FLAG_verbose) PrintVideoInfo(info, fileName);
}

static int StringToFourcc(const std::string &str) {
  union chint {
    int i;
    char c[4];
  };
  chint x = {0};
  for (int n = (str.size() < 4) ? (int)str.size() : 4; n--;) x.c[n] = str[n];
  return x.i;
}

struct FXApp {
  enum Err {
    errQuit = +1,                              // Application errors
    errFlag = +2,
    errRead = +3,
    errWrite = +4,
    errNone = NVCV_SUCCESS,                     // Video Effects SDK errors
    errGeneral = NVCV_ERR_GENERAL,
    errUnimplemented = NVCV_ERR_UNIMPLEMENTED,
    errMemory = NVCV_ERR_MEMORY,
    errEffect = NVCV_ERR_EFFECT,
    errSelector = NVCV_ERR_SELECTOR,
    errBuffer = NVCV_ERR_BUFFER,
    errParameter = NVCV_ERR_PARAMETER,
    errMismatch = NVCV_ERR_MISMATCH,
    errPixelFormat = NVCV_ERR_PIXELFORMAT,
    errModel = NVCV_ERR_MODEL,
    errLibrary = NVCV_ERR_LIBRARY,
    errInitialization = NVCV_ERR_INITIALIZATION,
    errFileNotFound = NVCV_ERR_FILE,
    errFeatureNotFound = NVCV_ERR_FEATURENOTFOUND,
    errMissingInput = NVCV_ERR_MISSINGINPUT,
    errResolution = NVCV_ERR_RESOLUTION,
    errUnsupportedGPU = NVCV_ERR_UNSUPPORTEDGPU,
    errWrongGPU = NVCV_ERR_WRONGGPU,
    errCudaMemory = NVCV_ERR_CUDA_MEMORY,       // CUDA errors
    errCudaValue = NVCV_ERR_CUDA_VALUE,
    errCudaPitch = NVCV_ERR_CUDA_PITCH,
    errCudaInit = NVCV_ERR_CUDA_INIT,
    errCudaLaunch = NVCV_ERR_CUDA_LAUNCH,
    errCudaKernel = NVCV_ERR_CUDA_KERNEL,
    errCudaDriver = NVCV_ERR_CUDA_DRIVER,
    errCudaUnsupported = NVCV_ERR_CUDA_UNSUPPORTED,
    errCudaIllegalAddress = NVCV_ERR_CUDA_ILLEGAL_ADDRESS,
    errCuda = NVCV_ERR_CUDA,
  };
  enum CompMode { compMatte, compLight, compGreen, compWhite, compNone, compBG, compBlur };

  FXApp() {
    _eff = nullptr;
    _bgblurEff = nullptr;
    _effectName = nullptr;
    _inited = false;
    _total = 0.0;
    _count = 0;
    _compMode = compLight;
    _showFPS = false;
    _stream = nullptr;
    _progress = false;
    _show = false;
    _framePeriod = 0.f;
    _lastTime = std::chrono::high_resolution_clock::time_point::min();
    _blurStrength = 0.5f;
    _maxInputWidth = 3840u;
    _maxInputHeight = 2160u;
    _maxNumberStreams = 1u;
    _batchOfStates = nullptr;
  }
  ~FXApp() {
    destroyEffect();
  }

  void setShow(bool show) { _show = show; }
  NvCV_Status createAigsEffect();
  void destroyEffect();
  NvCV_Status allocBuffers(unsigned width, unsigned height);
  NvCV_Status allocTempBuffers();
  Err processImage(const char *inFile, const char *outFile);
  Err processMovie(const char *inFile, const char *outFile);
  Err processKey(int key);
  void nextCompMode();
  void drawFrameRate(cv::Mat &img);
  Err appErrFromVfxStatus(NvCV_Status status) { return (Err)status; }
  const char *errorStringFromCode(Err code);

  NvVFX_Handle _eff, _bgblurEff;
  cv::Mat _srcImg;
  cv::Mat _dstImg;
  cv::Mat _bgImg;
  cv::Mat _resizedCroppedBgImg;
  NvCVImage _srcVFX;
  NvCVImage _dstVFX;
  bool _show;
  bool _inited;
  bool _showFPS;
  bool _progress;
  const char *_effectName;
  float _total;
  int _count;
  CompMode _compMode;
  float _framePeriod;
  CUstream _stream;
  std::chrono::high_resolution_clock::time_point _lastTime;
  NvCVImage _srcNvVFXImage;
  NvCVImage _dstNvVFXImage;
  NvCVImage _blurNvVFXImage;
  float _blurStrength;
  unsigned int _maxInputWidth;
  unsigned int _maxInputHeight;
  unsigned int _maxNumberStreams;
  std::vector<NvVFX_StateObjectHandle> _stateArray;
  NvVFX_StateObjectHandle* _batchOfStates;
};

const char *FXApp::errorStringFromCode(Err code) {
  struct LutEntry {
    Err code;
    const char *str;
  };
  static const LutEntry lut[] = {
      {errRead, "There was a problem reading a file"},
      {errWrite, "There was a problem writing a file"},
      {errQuit, "The user chose to quit the application"},
      {errFlag, "There was a problem with the command-line arguments"},
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
  } else {               // Ludicrous time interval; reset
    _framePeriod = 0.f;  // WAKE UP
  }
  _lastTime = now;
}

void FXApp::nextCompMode() {
  switch (_compMode) {
    default:
    case compMatte:
      _compMode = compLight;
      break;
    case compLight:
      _compMode = compGreen;
      break;
    case compGreen:
      _compMode = compWhite;
      break;
    case compWhite:
      _compMode = compNone;
      break;
    case compNone:
      _compMode = compBG;
      break;
    case compBG:
      _compMode = compBlur;
      break;
    case compBlur:
      _compMode = compMatte;
      break;
  }
}

FXApp::Err FXApp::processKey(int key) {
  static const int ESC_KEY = 27;
  switch (key) {
    case 'Q':
    case 'q':
    case ESC_KEY:
      return errQuit;
    case 'c':
    case 'C':
      nextCompMode();
      break;
    case 'f':
    case 'F':
      _showFPS = !_showFPS;
      break;
    case 'p':
    case 'P':
    case '%':
      _progress = !_progress;
      break;
    case 'm':
      _blurStrength += 0.05f;
      if (_blurStrength > 1.0) {
        _blurStrength = 1.0;
      }
      break;
    case 'n':
      _blurStrength -= 0.05f;
      if (_blurStrength < 0.0) {
        _blurStrength = 0.0;
      }
      break;
    default:
      break;
  }
  return errNone;
}

NvCV_Status FXApp::createAigsEffect() {
  NvCV_Status vfxErr;

  vfxErr = NvVFX_CreateEffect(NVVFX_FX_GREEN_SCREEN, &_eff);
  if (NVCV_SUCCESS != vfxErr) {
    std::cerr << "Error creating effect \"" << NVVFX_FX_GREEN_SCREEN << "\"\n";
    return vfxErr;
  }
  _effectName = NVVFX_FX_GREEN_SCREEN;

  if (!FLAG_modelDir.empty()) {
    vfxErr = NvVFX_SetString(_eff, NVVFX_MODEL_DIRECTORY, FLAG_modelDir.c_str());
  } 
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error setting the model path to \"" << FLAG_modelDir << "\"\n";
    return vfxErr;
  }

  const char *cstr;  // TODO: This is not necessary
  vfxErr = NvVFX_GetString(_eff, NVVFX_INFO, &cstr);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "AIGS modes not found \n" << std::endl;
    return vfxErr;
  }

  // Choose one mode -> set() -> Load() -> Run()
  vfxErr = NvVFX_SetU32(_eff, NVVFX_MODE, FLAG_mode);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error setting the mode \n";
    return vfxErr;
  }

  vfxErr = NvVFX_SetU32(_eff, NVVFX_CUDA_GRAPH, FLAG_cudaGraph?1u:0u);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error enabling cuda graph \n";
    return vfxErr;
  }

  vfxErr = NvVFX_CudaStreamCreate(&_stream);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error creating CUDA stream " << std::endl;
    return vfxErr;
  }

  vfxErr = NvVFX_SetCudaStream(_eff, NVVFX_CUDA_STREAM, _stream);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error setting up the cuda stream \n";
    return vfxErr;
  }

  // Set maximum width, height and number of streams and then call Load() again
  vfxErr = NvVFX_SetU32(_eff, NVVFX_MAX_INPUT_WIDTH, _maxInputWidth);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error setting the mode \n";
    return vfxErr;
  }

  vfxErr = NvVFX_SetU32(_eff, NVVFX_MAX_INPUT_HEIGHT, _maxInputHeight);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error setting the mode \n";
    return vfxErr;
  }

  vfxErr = NvVFX_SetU32(_eff, NVVFX_MAX_NUMBER_STREAMS, _maxNumberStreams);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error setting the mode \n";
    return vfxErr;
  }

  vfxErr = NvVFX_Load(_eff);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "Error loading the model \n";
    return vfxErr;
  }

  for (unsigned int i = 0; i < _maxNumberStreams; i++) {
    NvVFX_StateObjectHandle state;
    vfxErr = NvVFX_AllocateState(_eff, &state);
    if (NVCV_SUCCESS != vfxErr) {
      std::cerr << "Error allocate state variable for effect \"" << NVVFX_FX_GREEN_SCREEN << "\"\n";
      return vfxErr;
    }
    _stateArray.push_back(state);
  }

  // ------------------ create Background blur effect ------------------ //
  vfxErr = NvVFX_CreateEffect(NVVFX_FX_BGBLUR, &_bgblurEff);
  if (NVCV_SUCCESS != vfxErr) {
    std::cerr << "Error creating effect \"" << NVVFX_FX_BGBLUR << "\"\n";
    return vfxErr;
  }

  vfxErr = NvVFX_GetString(_bgblurEff, NVVFX_INFO, &cstr);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "BGBLUR info not found \n" << std::endl;
    return vfxErr;
  }

  vfxErr = NvVFX_SetCudaStream(_bgblurEff, NVVFX_CUDA_STREAM, _stream);
  if (vfxErr != NVCV_SUCCESS) {
    std::cerr << "BGBLUR error setting up the cuda stream \n";
    return vfxErr;
  }

  return vfxErr;
}

void FXApp::destroyEffect() {
  // If DeallocateState fails, all memory allocated in the SDK returns to the heap when the effect handle is destroyed.
  for (unsigned int i = 0; i < _stateArray.size(); i++) {
    NvVFX_DeallocateState(_eff, _stateArray[i]);
  }
  _stateArray.clear();

  if (_batchOfStates != nullptr) {
    free(_batchOfStates);
    _batchOfStates = nullptr;
  }

  NvVFX_DestroyEffect(_eff);
  _eff = nullptr;

  NvVFX_DestroyEffect(_bgblurEff);
  _bgblurEff = nullptr;

  if (_stream) {
    NvVFX_CudaStreamDestroy(_stream);
  }
}

static void overlay(const cv::Mat &image, const cv::Mat &mask, float alpha, cv::Mat &result) {
  cv::Mat maskClr;
  cv::cvtColor(mask, maskClr, cv::COLOR_GRAY2BGR);
  result = image * (1.f - alpha) + maskClr * alpha;
}

static NvCV_Status WriteRGBA(const NvCVImage *bgr, const NvCVImage *a, const std::string& name) {
  NvCV_Status err;
  NvCVImage bgra(bgr->width, bgr->height, NVCV_BGRA, NVCV_U8);
  NvCVImage aa(const_cast<NvCVImage*>(a), 0, 0, a->width, a->height);
  aa.pixelFormat = NVCV_A;  // This could be Y but we make sure it is interpreted as alpha
  err = NvCVImage_Transfer(bgr, &bgra, 0, 0, 0);   if (NVCV_SUCCESS != err) return err;
  err = NvCVImage_Transfer(&aa, &bgra, 0, 0, 0);   if (NVCV_SUCCESS != err) return err;
  cv::Mat ocv;
  CVWrapperForNvCVImage(&bgra, &ocv);
  return cv::imwrite(name, ocv) ? NVCV_SUCCESS : NVCV_ERR_WRITE;
}

FXApp::Err FXApp::processImage(const char *inFile, const char *outFile) {
  NvCV_Status vfxErr;
  bool ok;
  cv::Mat result;
  NvCVImage fxSrcChunkyGPU, fxDstChunkyGPU;

  // Allocate space for batchOfStates to hold state variable addresses
  // Assume that MODEL_BATCH Size is enough for this scenario
  unsigned int modelBatch = 1;
  BAIL_IF_ERR(vfxErr = NvVFX_GetU32(_eff, NVVFX_MODEL_BATCH, &modelBatch));
  _batchOfStates = (NvVFX_StateObjectHandle*) malloc(sizeof(NvVFX_StateObjectHandle) * modelBatch);
  if (_batchOfStates == nullptr) {
    vfxErr = NVCV_ERR_MEMORY;
    goto bail;
  }

  if (!_eff) return errEffect;
  _srcImg = cv::imread(inFile);
  if (!_srcImg.data) return errRead;

  _dstImg = cv::Mat::zeros(_srcImg.size(), CV_8UC1);
  if (!_dstImg.data) return errMemory;

  (void)NVWrapperForCVMat(&_srcImg, &_srcVFX);
  (void)NVWrapperForCVMat(&_dstImg, &_dstVFX);

  if (!fxSrcChunkyGPU.pixels)
  {
    BAIL_IF_ERR(vfxErr =
      NvCVImage_Alloc(&fxSrcChunkyGPU, _srcImg.cols, _srcImg.rows, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));
  }
  
  if (!fxDstChunkyGPU.pixels) {
    BAIL_IF_ERR(vfxErr =
        NvCVImage_Alloc(&fxDstChunkyGPU, _srcImg.cols, _srcImg.rows, NVCV_A, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));
  }

  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_INPUT_IMAGE, &fxSrcChunkyGPU));
  BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_OUTPUT_IMAGE, &fxDstChunkyGPU));
  BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_srcVFX, &fxSrcChunkyGPU, 1.0f, _stream, NULL));

  // Assign states from stateArray in batchOfStates
  // There is only one stream in this app
  _batchOfStates[0] = _stateArray[0];
  BAIL_IF_ERR(vfxErr = NvVFX_SetStateObjectHandleArray(_eff, NVVFX_STATE, _batchOfStates));

  BAIL_IF_ERR(vfxErr = NvVFX_Run(_eff, 0));
  BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&fxDstChunkyGPU, &_dstVFX, 1.0f, _stream, NULL));

  overlay(_srcImg, _dstImg, 0.5, result);
  if (!std::string(outFile).empty()) {
    if(IsLossyImageFile(outFile))
      fprintf(stderr, "WARNING: JPEG output file format will reduce image quality\n");
    vfxErr = WriteRGBA(&_srcVFX, &_dstVFX, outFile);
    if (NVCV_SUCCESS != vfxErr) {
      printf("%s: \"%s\"\n", NvCV_GetErrorStringFromCode(vfxErr), outFile);
      goto bail;
    }
    ok = cv::imwrite(std::string(outFile) + "_segmentation_mask.png", _dstImg);  // save segmentation mask too
    if (!ok) {
      printf("Error writing: \"%s_segmentation_mask.png\"\n", outFile);
      return errWrite;
    }
  }
  if (_show) {
    cv::imshow("Output", result);
    cv::waitKey(3000);
  }
bail:
  return (FXApp::Err)vfxErr;
}

FXApp::Err FXApp::processMovie(const char *inFile, const char *outFile) {
  float ms = 0.0f;
  FXApp::Err appErr = errNone;
  const int camIndex = 0;
  NvCV_Status vfxErr = NVCV_SUCCESS;
  bool ok;
  cv::Mat result;
  cv::VideoCapture reader;
  cv::VideoWriter writer;
  unsigned frameNum;
  VideoInfo info;
  unsigned int modelBatch = 1;

  if (inFile && !inFile[0]) inFile = nullptr;  // Set file paths to NULL if zero length
  if (outFile && !outFile[0]) outFile = nullptr;

  if (inFile) {
    reader.open(inFile);
  } else {
    reader.open(camIndex);
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
      if (camWidth) reader.set(cv::CAP_PROP_FRAME_WIDTH, camWidth);
      if (camHeight) reader.set(cv::CAP_PROP_FRAME_HEIGHT, camHeight);
    }
    printf("Camera frame: %.0f x %.0f\n", reader.get(cv::CAP_PROP_FRAME_WIDTH), reader.get(cv::CAP_PROP_FRAME_HEIGHT));
  }
  if (!reader.isOpened()) {
    if (!FLAG_webcam) printf("Error: Could not open video: \"%s\"\n", inFile);
    else              printf("Error: Webcam not found\n");
    return errRead;
  }

  GetVideoInfo(reader, (inFile ? inFile : "webcam"), &info);

  if (outFile) {
    ok = writer.open(outFile, StringToFourcc(FLAG_codec), info.frameRate, cv::Size(info.width, info.height));
    if (!ok) {
      printf("Cannot open \"%s\" for video writing\n", outFile);
      outFile = nullptr;
    }
  }

  unsigned int width = (unsigned int)reader.get(cv::CAP_PROP_FRAME_WIDTH);
  unsigned int height = (unsigned int)reader.get(cv::CAP_PROP_FRAME_HEIGHT);

  if (!FLAG_bgFile.empty())
  {
    _bgImg = cv::imread(FLAG_bgFile);
      if (!_bgImg.data)
      {
        return errRead;
      }
      else
      {
        // Find the scale to resize background such that image can fit into background
        float scale = float(height) / float(_bgImg.rows);
        if ((scale * _bgImg.cols) < float(width))
        {
          scale = float(width) / float(_bgImg.cols);
        }
        cv::Mat resizedBg;
        cv::resize(_bgImg, resizedBg, cv::Size(), scale, scale, cv::INTER_AREA);

        // Always crop from top left of background.
        cv::Rect rect(0, 0, width, height);
        _resizedCroppedBgImg = resizedBg(rect);
      }
  }

  // Allocate space for batchOfStates to hold state variable addresses
  // Assume that MODEL_BATCH Size is enough for this scenario
  BAIL_IF_ERR(vfxErr = NvVFX_GetU32(_eff, NVVFX_MODEL_BATCH, &modelBatch));
  _batchOfStates = (NvVFX_StateObjectHandle*) malloc(sizeof(NvVFX_StateObjectHandle) * modelBatch);
  if (_batchOfStates == nullptr) {
    vfxErr = NVCV_ERR_MEMORY;
    goto bail;
  }

  // allocate src for GPU
  if (!_srcNvVFXImage.pixels)
    BAIL_IF_ERR(vfxErr =
                    NvCVImage_Alloc(&_srcNvVFXImage, width, height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));

  // allocate dst for GPU
  if (!_dstNvVFXImage.pixels)
    BAIL_IF_ERR(vfxErr =
                    NvCVImage_Alloc(&_dstNvVFXImage, width, height, NVCV_A, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));

  // allocate blur for GPU
  if (!_blurNvVFXImage.pixels)
    BAIL_IF_ERR(vfxErr =
                    NvCVImage_Alloc(&_blurNvVFXImage, width, height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));

  for (frameNum = 0; reader.read(_srcImg); ++frameNum) {
    if (_srcImg.empty()) printf("Frame %u is empty\n", frameNum);

    _dstImg = cv::Mat::zeros(_srcImg.size(), CV_8UC1);  // TODO: Allocate and clear outside of the loop?
    BAIL_IF_NULL(_dstImg.data, vfxErr, NVCV_ERR_MEMORY);

    (void)NVWrapperForCVMat(&_srcImg, &_srcVFX);  // Ditto
    (void)NVWrapperForCVMat(&_dstImg, &_dstVFX);

    BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_INPUT_IMAGE, &_srcNvVFXImage));
    BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_OUTPUT_IMAGE, &_dstNvVFXImage));
    BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_srcVFX, &_srcNvVFXImage, 1.0f, _stream, NULL));

    // Assign states from stateArray in batchOfStates
    // There is only one stream in this app
    _batchOfStates[0] = _stateArray[0];
    BAIL_IF_ERR(vfxErr = NvVFX_SetStateObjectHandleArray(_eff, NVVFX_STATE, _batchOfStates));

    auto startTime = std::chrono::high_resolution_clock::now();
    BAIL_IF_ERR(vfxErr = NvVFX_Run(_eff, 0));
    auto endTime = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    _count += 1;
    if (_count > 0) {
      // skipping first frame
      _total += ms;
    }

    BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_dstNvVFXImage, &_dstVFX, 1.0f, _stream, NULL));

    result.create(_srcImg.rows, _srcImg.cols,
                  CV_8UC3);  // Make sure the result is allocated. TODO: allocate outsifde of the loop?
    BAIL_IF_NULL(result.data, vfxErr, NVCV_ERR_MEMORY);
    result.setTo(cv::Scalar::all(0));  // TODO: This may no longer be necessary since we no longer coerce to 16:9
    switch (_compMode) {
      case compNone:
        _srcImg.copyTo(result);
        break;
      case compBG: {
        if (FLAG_bgFile.empty())
        {
          _resizedCroppedBgImg = cv::Mat(_srcImg.rows, _srcImg.cols, CV_8UC3, cv::Scalar(118, 185, 0));
          size_t startX = _resizedCroppedBgImg.cols/20;
          size_t offsetY = _resizedCroppedBgImg.rows/20;
          std::string text = "No Background Image!";
          for (size_t startY = offsetY; startY < _resizedCroppedBgImg.rows; startY += offsetY)
          {
            cv::putText(_resizedCroppedBgImg, text, cv::Point(startX, startY),
                cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 0), 1);
          }
        }
        NvCVImage bgVFX;
        (void)NVWrapperForCVMat(&_resizedCroppedBgImg, &bgVFX);
        NvCVImage matVFX;
        (void)NVWrapperForCVMat(&result, &matVFX);
        NvCVImage_Composite(&_srcVFX, &bgVFX, &_dstVFX, &matVFX, _stream);
      } break;
      case compLight:
        if (inFile) {
          overlay(_srcImg, _dstImg, 0.5, result);
        } else {  // If the webcam was cropped, also crop the compositing
          cv::Rect rect(0, (_srcImg.rows - _srcVFX.height) / 2, _srcVFX.width, _srcVFX.height);
          cv::Mat subResult = result(rect);
          overlay(_srcImg(rect), _dstImg(rect), 0.5, subResult);
        }
        break;
      case compGreen: {
        const unsigned char bgColor[3] = {0, 255, 0};
        NvCVImage matVFX;
        (void)NVWrapperForCVMat(&result, &matVFX);
        NvCVImage_CompositeOverConstant(&_srcVFX, &_dstVFX, bgColor, &matVFX, _stream);
      } break;
      case compWhite: {
        const unsigned char bgColor[3] = {255, 255, 255};
        NvCVImage matVFX;
        (void)NVWrapperForCVMat(&result, &matVFX);
        NvCVImage_CompositeOverConstant(&_srcVFX, &_dstVFX, bgColor, &matVFX, _stream);
      } break;
      case compMatte:
        cv::cvtColor(_dstImg, result, cv::COLOR_GRAY2BGR);
        break;
      case compBlur:
        BAIL_IF_ERR(vfxErr = NvVFX_SetF32(_bgblurEff, NVVFX_STRENGTH, _blurStrength));
        BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_bgblurEff, NVVFX_INPUT_IMAGE_0, &_srcNvVFXImage));
        BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_bgblurEff, NVVFX_INPUT_IMAGE_1, &_dstNvVFXImage));
        BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_bgblurEff, NVVFX_OUTPUT_IMAGE, &_blurNvVFXImage));
        BAIL_IF_ERR(vfxErr = NvVFX_Load(_bgblurEff));
        BAIL_IF_ERR(vfxErr = NvVFX_Run(_bgblurEff, 0));

        NvCVImage matVFX;
        (void)NVWrapperForCVMat(&result, &matVFX);
        BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_blurNvVFXImage, &matVFX, 1.0f, _stream, NULL));

        break;
    }
    if (outFile) {
#define WRITE_COMPOSITE
#ifdef WRITE_COMPOSITE
      writer.write(result);
#else   // WRITE_MATTE
      writer.write(_dstImg);
#endif  // WRITE_MATTE
    }
    if (_show) {
      drawFrameRate(result);
      cv::imshow("Output", result);
      int key = cv::waitKey(1);
      if (key > 0) {
        appErr = processKey(key);
        if (errQuit == appErr) break;
      }
    }
    if (_progress)
        if(info.frameCount == 0)  // no progress for a webcam
            fprintf(stderr, "\b\b\b\b???%%");
        else
            fprintf(stderr, "\b\b\b\b%3.0f%%", 100.f * frameNum / info.frameCount);
  }

  if (_progress) fprintf(stderr, "\n");
  reader.release();
  if (outFile) writer.release();
bail:
  // Dealloc
  NvCVImage_Dealloc(&(_srcNvVFXImage));  // This is also called in the destructor, ...
  NvCVImage_Dealloc(&(_dstNvVFXImage));  // ... so is not necessary except in C code.
  NvCVImage_Dealloc(&(_blurNvVFXImage));
  return appErrFromVfxStatus(vfxErr);
}

// This path is used by nvVideoEffectsProxy.cpp to load the SDK dll
char *g_nvVFXSDKPath = NULL;

int chooseGPU() {
  // If the system has multiple supported GPUs then the application
  // should use CUDA driver APIs or CUDA runtime APIs to enumerate
  // the GPUs and select one based on the application's requirements

  // Cuda device 0
  return 0;
}

bool isCompModeEnumValid(const FXApp::CompMode& mode)
{
  if (mode != FXApp::CompMode::compMatte &&
      mode != FXApp::CompMode::compLight &&
      mode != FXApp::CompMode::compGreen &&
      mode != FXApp::CompMode::compWhite &&
      mode != FXApp::CompMode::compNone  &&
      mode != FXApp::CompMode::compBG    &&
      mode != FXApp::CompMode::compBlur)
    {
      return false;
    }
    return true;
}

int main(int argc, char **argv) {
  int nErrs = 0;
  nErrs = ParseMyArgs(argc, argv);
  if (nErrs) {
    Usage();
    return nErrs;
  }

  FXApp::Err fxErr = FXApp::errNone;
  FXApp app; 

  if (FLAG_inFile.empty() && !FLAG_webcam) {
    std::cerr << "Please specify --in_file=XXX or --webcam\n";
    ++nErrs;
  }
  if (FLAG_outFile.empty() && !FLAG_show) {
    std::cerr << "Please specify --out_file=XXX or --show\n";
    ++nErrs;
  }

  app._progress = FLAG_progress;
  app.setShow(FLAG_show);

  app._compMode = static_cast<FXApp::CompMode>(FLAG_compMode);
  if (!isCompModeEnumValid(app._compMode))
  {
    std::cerr << "Please specify a valid --comp_mode=XXX, valid range is [0,6] check help section\n";
    ++nErrs;
  }

  app._blurStrength = FLAG_blurStrength;
  if (app._blurStrength < 0) {
    app._blurStrength = 0;
  }
  else if (app._blurStrength > 1) {
    app._blurStrength = 1;
  }

  std::cout << "Processing " << FLAG_inFile << " mode " << FLAG_mode << " models " << FLAG_modelDir << std::endl;

  if (nErrs) {
    Usage();
    fxErr = FXApp::errFlag;
  } else {
    fxErr = app.appErrFromVfxStatus(app.createAigsEffect());
    if (FXApp::errNone == fxErr) {
      if (IsImageFile(FLAG_inFile.c_str()))
        fxErr = app.processImage(FLAG_inFile.c_str(), FLAG_outFile.c_str());
      else
        fxErr = app.processMovie(FLAG_inFile.c_str(), FLAG_outFile.c_str());
      if (fxErr == FXApp::errNone || fxErr == FXApp::errQuit) {
        fxErr = FXApp::errNone;  // Quitting isn't an error
        std::cout << "Processing time averaged over " << app._count << " runs is "
                  << ((float)app._total) / ((float)app._count - 1) << " ms. " << std::endl;
      }
    }
  }

  if (fxErr) std::cerr << "Error: " << app.errorStringFromCode(fxErr) << std::endl;
  return (int)fxErr;
}
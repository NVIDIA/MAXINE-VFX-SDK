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
#include <string>

#include "nvVideoEffects.h"

#ifdef _WIN32
#define _WINSOCKAPI_
#include <windows.h>
#include <tchar.h>
#else
#include <dlfcn.h>
typedef void* HMODULE;
typedef void* HANDLE;
typedef void* HINSTANCE;
#endif

// Parameter string does not include the file extension
#ifdef _WIN32
#define nvLoadLibrary(library) LoadLibrary(TEXT(library ".dll"))
#else
#define nvLoadLibrary(library) dlopen("lib" library ".so", RTLD_LAZY)
#endif


inline void* nvGetProcAddress(HINSTANCE handle, const char* proc) {
  if (nullptr == handle) return nullptr;
#ifdef _WIN32
  return GetProcAddress(handle, proc);
#else
  return dlsym(handle, proc);
#endif
}

inline int nvFreeLibrary(HINSTANCE handle) {
#ifdef _WIN32
  return FreeLibrary(handle);
#else
  return dlclose(handle);
#endif
}

HINSTANCE getNvVfxLib() {

  TCHAR path[MAX_PATH], fullPath[MAX_PATH];

  // There can be multiple apps on the system,
  // some might include the SDK in the app package and
  // others might expect the SDK to be installed in Program Files
  GetEnvironmentVariable(TEXT("NV_VIDEO_EFFECTS_PATH"), path, MAX_PATH);
  if (_tcscmp(path, TEXT("USE_APP_PATH"))) {
    // App has not set environment variable to "USE_APP_PATH"
    // So pick up the SDK dll and dependencies from Program Files
    GetEnvironmentVariable(TEXT("ProgramFiles"), path, MAX_PATH);
    size_t max_len = sizeof(fullPath)/sizeof(TCHAR);
    _stprintf_s(fullPath, max_len, TEXT("%s\\NVIDIA Corporation\\NVIDIA Video Effects\\"), path);
    SetDllDirectory(fullPath);
  }
  static const HINSTANCE NvVfxLib = nvLoadLibrary("NVVideoEffects");
  return NvVfxLib;
}

NvCV_Status NvVFX_API NvVFX_GetVersion(unsigned int* version) {
  static const auto funcPtr = (decltype(NvVFX_GetVersion)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetVersion");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(version);
}

NvCV_Status NvVFX_API NvCVImage_Init(NvCVImage* im, unsigned width, unsigned height, int pitch, void* pixels,
                                       NvCVImage_PixelFormat format, NvCVImage_ComponentType type, unsigned isPlanar,
                                       unsigned onGPU) {
  static const auto funcPtr = (decltype(NvCVImage_Init)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_Init");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(im, width, height, pitch, pixels, format, type, isPlanar, onGPU);
}

void NvVFX_API NvCVImage_InitView(NvCVImage* subImg, NvCVImage* fullImg, int x, int y, unsigned width,
                                   unsigned height) {
  static const auto funcPtr = (decltype(NvCVImage_InitView)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_InitView");

  if (nullptr != funcPtr) funcPtr(subImg, fullImg, x, y, width, height);
}

NvCV_Status NvCVImage_Alloc(NvCVImage* im, unsigned width, unsigned height, NvCVImage_PixelFormat format,
                              NvCVImage_ComponentType type, unsigned isPlanar, unsigned onGPU, unsigned alignment) {
  static const auto funcPtr = (decltype(NvCVImage_Alloc)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_Alloc");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(im, width, height, format, type, isPlanar, onGPU, alignment);
}

NvCV_Status NvVFX_API NvCVImage_Realloc(NvCVImage* im, unsigned width, unsigned height,
                                          NvCVImage_PixelFormat format, NvCVImage_ComponentType type,
                                          unsigned isPlanar, unsigned onGPU, unsigned alignment) {
  static const auto funcPtr = (decltype(NvCVImage_Realloc)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_Realloc");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(im, width, height, format, type, isPlanar, onGPU, alignment);
}

void NvVFX_API NvCVImage_Dealloc(NvCVImage* im) {
  static const auto funcPtr = (decltype(NvCVImage_Dealloc)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_Dealloc");

  if (nullptr != funcPtr) funcPtr(im);
}

NvCV_Status NvVFX_API NvCVImage_Create(unsigned width, unsigned height, NvCVImage_PixelFormat format,
                                         NvCVImage_ComponentType type, unsigned isPlanar, unsigned onGPU,
                                         unsigned alignment, NvCVImage** out) {
  static const auto funcPtr = (decltype(NvCVImage_Create)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_Create");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(width, height, format, type, isPlanar, onGPU, alignment, out);
}

void NvVFX_API NvCVImage_Destroy(NvCVImage* im) {
  static const auto funcPtr = (decltype(NvCVImage_Destroy)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_Destroy");

  if (nullptr != funcPtr) funcPtr(im);
}

void NvVFX_API NvCVImage_ComponentOffsets(NvCVImage_PixelFormat format, int* rOff, int* gOff, int* bOff, int* aOff,
                                           int* yOff) {
  static const auto funcPtr =
      (decltype(NvCVImage_ComponentOffsets)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_ComponentOffsets");

  if (nullptr != funcPtr) funcPtr(format, rOff, gOff, bOff, aOff, yOff);
}

NvCV_Status NvVFX_API NvCVImage_Transfer(const NvCVImage* src, NvCVImage* dst, float scale, CUstream_st* stream,
                                           NvCVImage* tmp) {
  static const auto funcPtr = (decltype(NvCVImage_Transfer)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_Transfer");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(src, dst, scale, stream, tmp);
}

NvCV_Status NvVFX_API NvCVImage_Composite(const NvCVImage* fg, const NvCVImage* bg, const NvCVImage* mat, NvCVImage* dst) {
  static const auto funcPtr = (decltype(NvCVImage_Composite)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_Composite");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(fg, bg, mat, dst);
}

NvCV_Status NvVFX_API NvCVImage_CompositeOverConstant(const NvCVImage* src, const NvCVImage* mat,
                                                        const unsigned char bgColor[3], NvCVImage* dst) {
  static const auto funcPtr =
      (decltype(NvCVImage_CompositeOverConstant)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_CompositeOverConstant");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(src, mat, bgColor, dst);
}

NvCV_Status NvVFX_API NvCVImage_FlipY(const NvCVImage* src, NvCVImage* dst) {
  static const auto funcPtr = (decltype(NvCVImage_FlipY)*)nvGetProcAddress(getNvVfxLib(), "NvCVImage_FlipY");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(src, dst);
}

NvCV_Status NvVFX_API NvVFX_CreateEffect(NvVFX_EffectSelector code, NvVFX_Handle* obj) {
  static const auto funcPtr = (decltype(NvVFX_CreateEffect)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_CreateEffect");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(code, obj);
}

void NvVFX_API NvVFX_DestroyEffect(NvVFX_Handle obj) {
  static const auto funcPtr = (decltype(NvVFX_DestroyEffect)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_DestroyEffect");

  if (nullptr != funcPtr) funcPtr(obj);
}

NvCV_Status NvVFX_API NvVFX_SetU32(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, unsigned int val) {
  static const auto funcPtr = (decltype(NvVFX_SetU32)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetU32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_SetS32(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, int val) {
  static const auto funcPtr = (decltype(NvVFX_SetS32)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetS32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_SetF32(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, float val) {
  static const auto funcPtr = (decltype(NvVFX_SetF32)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetF32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_SetF64(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, double val) {
  static const auto funcPtr = (decltype(NvVFX_SetF64)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetF64");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_SetU64(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, unsigned long long val) {
  static const auto funcPtr = (decltype(NvVFX_SetU64)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetU64");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_SetImage(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, NvCVImage* im) {
  static const auto funcPtr = (decltype(NvVFX_SetImage)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetImage");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, im);
}

NvCV_Status NvVFX_API NvVFX_SetObject(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, void* ptr) {
  static const auto funcPtr = (decltype(NvVFX_SetObject)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetObject");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, ptr);
}

NvCV_Status NvVFX_API NvVFX_SetString(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, const char* str) {
  static const auto funcPtr = (decltype(NvVFX_SetString)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetString");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, str);
}

NvCV_Status NvVFX_API NvVFX_SetCudaStream(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, CUstream stream) {
  static const auto funcPtr = (decltype(NvVFX_SetCudaStream)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_SetCudaStream");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, stream);
}

NvCV_Status NvVFX_API NvVFX_GetU32(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, unsigned int* val) {
  static const auto funcPtr = (decltype(NvVFX_GetU32)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetU32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_GetS32(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, int* val) {
  static const auto funcPtr = (decltype(NvVFX_GetS32)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetS32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_GetF32(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, float* val) {
  static const auto funcPtr = (decltype(NvVFX_GetF32)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetF32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_GetF64(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, double* val) {
  static const auto funcPtr = (decltype(NvVFX_GetF64)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetF64");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_GetU64(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, unsigned long long* val) {
  static const auto funcPtr = (decltype(NvVFX_GetU64)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetU64");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, val);
}

NvCV_Status NvVFX_API NvVFX_GetImage(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, NvCVImage* im) {
  static const auto funcPtr = (decltype(NvVFX_GetImage)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetImage");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, im);
}

NvCV_Status NvVFX_API NvVFX_GetObject(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, void** ptr) {
  static const auto funcPtr = (decltype(NvVFX_GetObject)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetObject");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, ptr);
}

NvCV_Status NvVFX_API NvVFX_GetString(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, const char** str) {
  static const auto funcPtr = (decltype(NvVFX_GetString)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetString");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, str);
}

NvCV_Status NvVFX_API NvVFX_GetCudaStream(NvVFX_Handle obj, NvVFX_ParameterSelector paramName, CUstream* stream) {
  static const auto funcPtr = (decltype(NvVFX_GetCudaStream)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_GetCudaStream");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, paramName, stream);
}

NvCV_Status NvVFX_API NvVFX_Run(NvVFX_Handle obj, int async) {
  static const auto funcPtr = (decltype(NvVFX_Run)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_Run");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj, async);
}

NvCV_Status NvVFX_API NvVFX_Load(NvVFX_Handle obj) {
  static const auto funcPtr = (decltype(NvVFX_Load)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_Load");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(obj);
}

NvCV_Status NvVFX_API NvVFX_CudaStreamCreate(CUstream* stream) {
  static const auto funcPtr =
      (decltype(NvVFX_CudaStreamCreate)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_CudaStreamCreate");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(stream);
}

NvCV_Status NvVFX_API NvVFX_CudaStreamDestroy(CUstream stream) {
  static const auto funcPtr =
      (decltype(NvVFX_CudaStreamDestroy)*)nvGetProcAddress(getNvVfxLib(), "NvVFX_CudaStreamDestroy");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(stream);
}

#ifdef _WIN32
__declspec(dllexport) const char* __cdecl
#else
const char*
#endif  // _WIN32 or linux
    NvCV_GetErrorStringFromCode(NvCV_Status code) {
  static const auto funcPtr =
      (decltype(NvCV_GetErrorStringFromCode)*)nvGetProcAddress(getNvVfxLib(), "NvCV_GetErrorStringFromCode");

  if (nullptr == funcPtr) return "Cannot find NVVideoEffects DLL or its dependencies";
  return funcPtr(code);
}

# README
## NVIDIA MAXINE VideoEffects SDK: API Source Code and Sample Applications

NVIDIA MAXINE VideoEffects SDK is an SDK for enhancing and applying filters to videos at real-time. The SDK is powered by NVIDIA graphics processing units (GPUs) with Tensor Cores, and as a result, the algorithm throughput is greatly accelerated, and latency is reduced.

The SDK has the following AI features:

- **AI Green Screen**, which segments and masks the background areas in a video or image.
- **Background Blur**, which uses the segmentation mask from the AI Green Screen filter or other sources, and produces a blur effect over the background of a video or iamge.
- **Encoder Artifact Reduction**, which reduces the blocky and noisy artifacts from an encoded video while preserving the details of the original video.
- **Super Resolution**, which upscales a video while also reducing the blocky and noisy artifacts. It can enhance the details and sharpen the output while simultaneously preserving the content. This is suitable for upscaling lossy content.
- **Upscale**, which is a very fast and light-weight method for upscaling an input video. It also provides a sharpening parameter to sharpen the resulting output. This feature can be optionally pipelined with the encoder artifact reduction feature to enhance the scale while reducing the video artifacts.
- **Webcam Denoising**, which removes noise from a webcam video while preserving the texture details.

<p align="center">
<img src="https://github.com/NVIDIA/MAXINE-VFX-SDK/blob/master/resources/SR.gif" alt="NVIDIA Super Resolution" width="640" height="320"/>
 </p>

<p align="center">
<img src="https://github.com/NVIDIA/MAXINE-VFX-SDK/blob/master/resources/Denoise.gif" alt="NVIDIA Webcam Denoising" width="640" height="320"/>
 </p>

The SDK provides several sample applications that demonstrate the features listed above in real time by using offline videos.
- **AI Green Screen App**, which is a sample app that demonstrates the background segmentation feature.
- **VideoEffects App**, which is a sample app that can invoke each of Encoder Artifact Reduction, Super Resolution or Upscale features individually.
- **UpscalePipeline App**, which is a sample app that pipelines the Encoder Artifact Reduction feature with the Upscale feature.
- **DenoiseEffect App**, which is a sample app that demonstrates the webcam denoising feature.
 
The input and output resolutions supported by the features of the SDK are listed below.
- The Encoder Artifact Reduction feature supports between 90p to 1080p as input resolutions. 
- The Super Resolution feature supports between 90p to 2160p as input resolutions.
   - Super Resolution supports the following scaling factors: 4/3x (~1.33x), 1.5x, 2x, 3x and 4x.
   - 2160p input is only supported for the following scaling factors: 4/3x (~1.33x), 1.5x and 2x
   - The maximum output resolution for the Super Resolution feature is 4320p.
- The Upscale feature supports any input resolution, and the following scaling factors: 4/3x (~1.33x), 1.5x, 2x, 3x  and 4x.
- The Webcam Denoising feature supports between 80p to 1080p as input resolutions.
- The AI Green Screen and Background Blur features require that an input image/video be at least 288 pixels high.

NVIDIA MAXINE VideoEffects SDK is distributed in the following parts:

- This open source repository that includes the [SDK API and proxy linking source code](https://github.com/NVIDIA/MAXINE-VFX-SDK/tree/master/nvvfx), and [sample applications and their dependency libraries](https://github.com/NVIDIA/MAXINE-VFX-SDK/tree/master/samples).
- An installer hosted on [NVIDIA Maxine End-user Redistributables page](https://www.nvidia.com/broadcast-sdk-resources) that installs the SDK DLLs, the models, and the SDK dependency libraries.

Please refer to the [SDK System guide](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-system-guide/) for configuring and integrating the SDK, compiling and running the sample applications. Please visit the [NVIDIA MAXINE Video Effects SDK](https://developer.nvidia.com/maxine-getting-started) webpage for more information about the SDK.

## System requirements
The SDK is supported on NVIDIA GPUs that are based on the NVIDIA® Turing™ or Ampere™ architecture and have Tensor Cores.

* Windows OS supported: 64-bit Windows 10 or later
* Microsoft Visual Studio: 2017 (MSVC15.0) or later
* CMake: v3.12 or later
* NVIDIA Graphics Driver for Windows: 465.89 or later

## NVIDIA MAXINE Branding Guidelines
If you integrate an NVIDIA MAXINE SDK within your product, please follow the required branding guidelines that are available [here](https://www.nvidia.com/maxine-sdk-guidelines/)

## Compiling the sample apps

### Steps

The open source repository includes the source code to build the sample applications, and a proxy file NVVideoEffectsProxy.cpp to enable compilation without explicitly linking against the SDK DLL.

**Note: To download the models and runtime dependencies required by the features, you need to run the [SDK Installer](https://www.nvidia.com/broadcast-sdk-resources).**

1.	In the root folder of the downloaded source code, start the CMake GUI and specify the source folder and a build folder for the binary files.
*	For the source folder, ensure that the path ends in OSS.
*	For the build folder, ensure that the path ends in OSS/build.
2.  Use CMake to configure and generate the Visual Studio solution file.
*	Click Configure.
*	When prompted to confirm that CMake can create the build folder, click OK.
*	Select Visual Studio for the generator and x64 for the platform.
*	To complete configuring the Visual Studio solution file, click Finish.
*	To generate the Visual Studio Solution file, click Generate.
*	Verify that the build folder contains the NvVideoEffects_SDK.sln file.
3.  Use Visual Studio to generate the application binary .exe file from the NvVideoEffects_SDK.sln file.
*	In CMake, to open Visual Studio, click Open Project.
*	In Visual Studio, select Build > Build Solution.

## Documentation
Please refer to the online documentation guides -
* [NVIDIA Video Effects SDK Programming Guide](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html)
* [NVIDIA Video Effects SDK System Guide](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-system-guide/index.html)
* [NvCVImage API Guide](https://docs.nvidia.com/deeplearning/maxine/nvcvimage-api-guide/index.html)

PDF versions of these guides are also available at the following locations - 
* [NVIDIA Video Effects SDK Programming Guide](https://docs.nvidia.com/deeplearning/maxine/pdf/vfx-sdk-programming-guide.pdf)
* [NVIDIA Video Effects SDK System Guide](https://docs.nvidia.com/deeplearning/maxine/pdf/vfx-sdk-system-guide.pdf)
* [NvCVImage API Guide](https://docs.nvidia.com/deeplearning/maxine/pdf/nvcvimage-api-guide.pdf)

# README
## NVIDIA VideoEffects SDK: API Source Code and Sample Applications

NVIDIA VideoEffects SDK is an SDK for enhancing and applying filters to videos at real-time. The SDK is powered by NVIDIA graphics processing units (GPUs) with Tensor Cores, and as a result, the algorithm throughput is greatly accelerated, and latency is reduced.

NVIDIA VideoEffects SDK has the following AI features:

- **Encoder Artifact Reduction**, which reduces the blocky and noisy artifacts from an encoded video while preserving the details of the original video.
- **Super Resolution**, which upscales a video while also reducing the blocky and noisy artifacts. It can enhance the details and sharpen the output while simultaneously preserving the content. This is suitable for upscaling lossy content.
- **Upscale**, which is a very fast and light-weight method for upscaling an input video. It also provides a sharpening parameter to sharpen the resulting output. This feature can be optionally pipelined with the encoder artifact reduction feature to enhance the scale while reducing the video artifacts.

NVIDIA VideoEffects SDK provides two sample applications that demonstrate the features listed above in real time by using offline videos.
- **VideoEffects App**, which is a sample app that can invoke each feature individually.
- **UpscalePipeline App**, which is a sample app that pipelines the Encoder Artifact Reduction feature with the Upscale feature.
 
All features in the VideoEffects SDK support 720p and 1080p as input resolutions. These are the scaling factors supported by the Super Resolution feature:
- **720p inputs** can be scaled by a factor of 1.5x or 2x.
- **1080p inputs** can be scaled by a factor of 1.33x or 2x.

Additionally, the Upscale feature supports any input resolution, and the following scaling factors:
- 1.33x, 1.5x, 2x or 3x  

NVIDIA VideoEffects SDK is distributed in the following parts:

- This open source repository that includes the [SDK API and proxy linking source code](https://github.com/NVIDIA/BROADCAST-VFX-SDK/tree/master/nvvfx), and [sample applications and their dependency libraries](https://github.com/NVIDIA/BROADCAST-VFX-SDK/tree/master/samples).
- An installer hosted on [RTX broadcast engine developer page](https://developer.nvidia.com/rtx-broadcast-engine) that installs the SDK DLLs, the models, and the SDK dependency libraries.

Please refer to [SDK programming guide](https://github.com/NVIDIA/BROADCAST-VFX-SDK/blob/master/docs/NVIDIA%20Video%20Effects%20SDK%20Programming%20Guide.pdf) for configuring and integrating the SDK, compiling and running the sample applications.

## System requirements
The SDK is supported on NVIDIA GPUs that are based on the NVIDIA® Turing™ architecture. Although the SDK can run on Turing™ GPUs without Tensor Cores, it is optimized for much higher performance on GPUs with Tensor Cores.

* Windows OS supported: 64-bit Windows 10
* Microsoft Visual Studio: 2017 (MSVC15.0) or later
* CMake: v3.12 or later
* NVIDIA Graphics Driver for Windows: 455.57 or later

## NVIDIA Branding Guidelines
If you integrate an NVIDIA Broadcast Engine SDK within your product, please follow the required branding guidelines that are available [here](
https://nvidia.frontify.com/d/uAobRitG8H8B)

## Compiling the sample apps

### Steps

The open source repository includes the source code to build the sample applications, and a proxy file NVVideoEffectsProxy.cpp to enable compilation without explicitly linking against the SDK DLL.

**Note: To download the models and runtime dependencies required by the features, you need to run the [SDK Installer](https://developer.nvidia.com/rtx-broadcast-engine).**

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

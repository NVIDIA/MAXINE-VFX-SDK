SETLOCAL
SET PATH=%PATH%;..\external\opencv\bin;
REM Use --show to show the output in a window or use --out_file=<filename> to write output to file
UpscalePipelineApp.exe --in_file=..\input\input1.jpg --ar_strength=0 --upscale_strength=0 --resolution=1080 --show --out_file=ar_sr_0.png
UpscalePipelineApp.exe --in_file=..\input\input1.jpg --ar_strength=0 --upscale_strength=1 --resolution=1080 --show --out_file=ar_sr_1.png


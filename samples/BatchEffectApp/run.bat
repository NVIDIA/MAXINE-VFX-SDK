SETLOCAL
SET PATH=%PATH%;..\external\opencv\bin;
SET IMAGE_LIST=..\input\LeFret_000900.jpg ..\input\LeFret_001400.jpg ..\input\LeFret_003400.jpg ..\input\LeFret_012300.jpg
BatchEffectApp.exe --effect=ArtifactReduction --out_file=ArtifactReduction_%%04u.png %IMAGE_LIST%
BatchEffectApp.exe --effect=SuperRes --out_file=SuperRes_%%04u.png --scale=1.5 %IMAGE_LIST%
BatchEffectApp.exe --effect=Upscale --out_file=Upscale_%%04u.png --scale=1.5 %IMAGE_LIST% 

SET VIDEO_LIST=..\input\input_0_100_frames.mp4 ..\input\input_100_200_frames.mp4
BatchAigsEffectApp.exe --out_file=GreenScreen_%04u.mp4 %VIDEO_LIST%
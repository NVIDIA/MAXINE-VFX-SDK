SETLOCAL
SET PATH=%PATH%;..\external\opencv\bin;
DenoiseEffectApp.exe --webcam --strength=0 --show
DenoiseEffectApp.exe --webcam --strength=1 --show
SETLOCAL
SET PATH=%PATH%;..\external\opencv\bin;
REM Use --show to show the output in a window or use --out_file=<filename> to write output to file
VideoEffectsApp.exe --in_file=..\input\input1.jpg --out_file=ar_1.png --effect=ArtifactReduction --mode=1 --show
VideoEffectsApp.exe --in_file=..\input\input1.jpg --out_file=ar_0.png --effect=ArtifactReduction --mode=0 --show
VideoEffectsApp.exe --in_file=..\input\input2.jpg --out_file=sr_0.png --effect=SuperRes --resolution=2160 --mode=0 --show
VideoEffectsApp.exe --in_file=..\input\input2.jpg --out_file=sr_1.png --effect=SuperRes --resolution=2160 --mode=1 --show
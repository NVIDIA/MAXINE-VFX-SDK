SETLOCAL
SET PATH=%PATH%;..\external\opencv\bin;
REM Use --show to show the output in a window or use --out_file=<filename> to write output to file
VideoEffectsApp.exe --in_file=..\input\input1.jpg --out_file=ar_1.png --effect=ArtifactReduction --strength=1 --show
VideoEffectsApp.exe --in_file=..\input\input1.jpg --out_file=ar_0.png --effect=ArtifactReduction --strength=0 --show
VideoEffectsApp.exe --in_file=..\input\input2.png --out_file=sr_0.png --effect=SuperRes --resolution=1080 --strength=0 --show
VideoEffectsApp.exe --in_file=..\input\input2.png --out_file=sr_1.png --effect=SuperRes --resolution=1080 --strength=1 --show
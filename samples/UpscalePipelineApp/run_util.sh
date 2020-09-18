# --- do not invoke this directly ---
#
# this is a helper script for run_VFX_*.sh to set up
# PATH and LD_LIBRARY_PATH automatically
#

findpath() {
    # findpath "descriptive error text" file-that-must-exist locations-to-look...
    # used to find a library based on files that should exist in that location
    err_desc="$1"
    req_file="$2"
    shift 2
    for prefix in "$@"
    do
        if [ -e "$prefix/$req_file" ]
        then
            echo $prefix
            return
        fi
    done
    echo "Could not find $err_desc" 1>&2
    exit 1
}


_CUDA=`findpath "Cuda 10.1/10.2"            include/cuda.h    $CUDA_TOOLKIT /usr/local/cuda-10.1 /usr/local/cuda-10.2`
_TRT=`findpath  "TensorRT 6.0.1.5/6.0.1.8"  include/NvInfer.h $TensorRT_DIR /usr/local/TensorRT-6.0.1.5 /usr/local/TensorRT-6.0.1.8`
_VFX=`findpath  "VideoFX" lib/libNVVideoEffects.so ../lib /usr/local/VideoFX /usr`

export LD_LIBRARY_PATH=$_VFX/lib:$_CUDA/lib64:$_TRT/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

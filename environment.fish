
#
#  configure here
#

set cuda /usr/local/cuda-11.1
set conda $CONDA_PREFIX/envs/ryn

#
# --------------------
#

echo
echo '-------------------------'
echo '          R Y N'
echo '-------------------------'
echo


function _abort
    echo 'abort!'
    exit 2
end

function _prompt
    while true
        read -p 'echo "alright?" [yn] ' -l answer

        switch "$answer"
            case Y y
                return 0
            case '' N n
                return 1
        end
    end
end


echo 'setting enviroment variables'

echo
echo set -x CUDA_HOME "$cuda"
echo set -x PATH "$cuda/bin" $PATH
echo set -x HOROVOD_CUDA_HOME "$cuda"
echo set -x HOROVOD_NCCL_HOME "$conda"
echo set -x HOROVOD_GPU_ALLREDUCE "NCCL"
echo set -x HOROVOD_GPU_BROADCAST "NCCL"
echo set -x LD_LIBRARY_PATH "$cuda/lib64" $LD_LIBRARY_PATH
echo


if not _prompt
    _abort
end

set -x CUDA_HOME "$cuda"
set -x PATH "$cuda/bin" $PATH
set -x HOROVOD_CUDA_HOME "$cuda"
set -x HOROVOD_NCCL_HOME "$conda"
set -x HOROVOD_NCCL_LINK "SHARED"
set -x HOROVOD_GPU_ALLREDUCE "NCCL"
set -x HOROVOD_GPU_BROADCAST "NCCL"
set -x HOROVOD_WITH_PYTORCH 1
set -x LD_LIBRARY_PATH "$cuda/lib64" $LD_LIBRARY_PATH


echo 'creating conda environment'
conda env create -f environment.yml --force
or _abort

exit

echo 'activating environment'
conda activate ryn
or _abort

echo 'checking horovod'
horovodrun --check-build
or _abort

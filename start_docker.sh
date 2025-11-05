WORKDIR="/workspace"
WORKSPACE_DIR="${PWD}"
# IMAGE_NAME="rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.7.1"
docker load -i ../docker_images/tracelense_auto.tar
IMAGE_NAME="tracelense_auto"
CONTAINER_NAME="tracelense_auto_v2"
docker run \
    -it --rm \
    --network=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 320G \
    --workdir $WORKDIR \
    -v $WORKSPACE_DIR:$WORKDIR \
    --entrypoint /bin/bash \
    --name $CONTAINER_NAME \
    $IMAGE_NAME
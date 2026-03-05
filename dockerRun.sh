docker build -t nfc-single-round-closedform-regression-vbm-dev -f Dockerfile-dev .

docker run --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --name nfc-single-round-closedform-regression-vbm \
    -v /$(pwd):/workspace \
    -w //workspace \
    nfc-single-round-closedform-regression-vbm-dev:latest
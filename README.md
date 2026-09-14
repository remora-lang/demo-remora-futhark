# Demo: Compiling Remora with the Futhark compiler

This demo demonstrates how to benchmark a Remora implementation of YOLOv4 that
is ultimately compiled with the Futhark compiler. It is intended as an interim
steep until the [MLIR backend](https://github.com/remora-lang/mlir-backend)
improves.

All of the generated code here has been automatically produced with the various
tools built as part of the Remora project, but is embedded in the repository to
faciliate the demonstration.

The Futhark side is a little hacky, as its benchmarking tooling was not expected
to be used in this way.

## Usage

Download `yolov4_input.bin` and `yolov4_expected.bin`, which are Futhark data
files produced from
[input.bin](https://github.com/remora-lang/remora/blob/4150daee8754d230da3a358c3320c821011a6b68/examples/darknet/input.bin)
and
[yolov4.weights](https://github.com/remora-lang/remora/blob/main/examples/darknet/yolov4.weights):

```
$ curl -O https://sigkill.dk/junk/yolov4_input.bin
$ curl -O https://sigkill.dk/junk/yolov4_expected.bin
```

### Benchmarking with sequential C backend

*Very slow.*

```
$ futhark dev --seq-mem --backend=c --server yolov4.fut_soacs
$ futhark bench --skip-compilation yolov4.fut
```

### Benchmarking with CUDA backend

```
$ futhark dev --gpu-mem --backend=cuda --server yolov4.fut_soacs
$ futhark bench --skip-compilation yolov4.fut
```


## Pipeline

The original program is [yolov4.remora](./yolov4.remora), slightly modified
compared to the upstream one by changing some `@reduce` to `@reduce/zero`.

It was compiled to Futhark as follows:

```
$ remora futhark < yolov4.remora > yolov4.fut_soacs
```

The `yolov4_input.bin` file has been produced by the script
[`mk_yolov4_input.py`](./mk_yolov4_input.py). It simply transforms the YOLOv4
input and output data such that it can be understood by Futhark's benchmarking
tools. With the YOLO data files in the current directory, run:

```
$ python3 mk_yolov4_input.py
```

#!/bin/bash

python3 -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --grpc_python_out=. \
  src/grpc_server/evacuai_rl.proto

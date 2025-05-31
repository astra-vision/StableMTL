#!/bin/bash

# Set these variables to your own paths
export CODE_DIR=$(pwd)
export BASE_DIR=${BASE_DIR:-$SCRATCH}
export PREPROCESSED_DIR=${PREPROCESSED_DIR:-"$BASE_DIR/preprocessed"}
export RAW_DATA_DIR=${RAW_DATA_DIR:-"$BASE_DIR/raw_data"}
export BASE_CKPT_DIR=${BASE_CKPT_DIR:-"$CODE_DIR/checkpoints"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$BASE_DIR/stablemtl_output"}

# Create directories if they don't exist
mkdir -p $PREPROCESSED_DIR
mkdir -p $RAW_DATA_DIR
mkdir -p $BASE_CKPT_DIR
mkdir -p $OUTPUT_DIR

# Print current environment settings
echo "StableMTL environment settings:"
echo "CODE_DIR: $CODE_DIR"
echo "BASE_DIR: $BASE_DIR"
echo "PREPROCESSED_DIR: $PREPROCESSED_DIR"
echo "RAW_DATA_DIR: $RAW_DATA_DIR"
echo "BASE_CKPT_DIR: $BASE_CKPT_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"

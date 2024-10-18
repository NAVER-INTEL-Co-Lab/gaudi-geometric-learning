# Gaudi-v2: For geometric deep learning


This repository is designed for implementing and testing geometric deep learning functionalities and algorithms using Intel Gaudi-v2 devices.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Content](#contents)

## Introduction
Geometric deep learning is a rapidly growing field that focuses on extending deep learning techniques to non-Euclidean domains such as graphs.
This repository aims to bridge the gap by providing support for Intel Gaudi-v2 devices, addressing the challenge that existing libraries like PyG (PyTorch Geometric) and DGL (Deep Graph Library) only support CUDA GPUs.

## Features
- Implementation of geometric deep learning functionalities and models on Intel Gaudi-v2 devices.
- Code adaptation tools Intel Gaudi-v2 devices.
- Benchmarking and performance analysis tools.
- Comprehensive documentation and examples.

## Contents
Below are the detailed contents of this library.

### Raw-PyTorch implementation of torch-sparse functionalities
- [x] `scatter_sum` / `scatter_add`
- [x] `scatter_mean`
- [x] `scatter_max` / `scatter_min`
- [x] `sactter_mul`
- [x] `scatter_log_softmax`
- [x] `scatter_logsumexp`
- [x] `scatter_std`

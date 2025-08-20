// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Sample TTIR: Simple Linear Layer
// This demonstrates a basic matrix multiplication (linear layer) in TTIR format

module {
  func.func @simple_linear(%input: tensor<1x256xf32>) -> tensor<1x128xf32> {
    %weights = ttir.empty() : tensor<256x128xf32>
    %result = "ttir.generic"(%input, %weights) <{operation = "matmul"}> : (tensor<1x256xf32>, tensor<256x128xf32>) -> tensor<1x128xf32>
    return %result : tensor<1x128xf32>
  }
} 
#system_desc = #ttcore.system_desc<{chipIds = [0], chipDescs = [{arch = #ttcore.chip_arch<wormhole_b0>, grid = #ttcore.grid<8 x 10>, l1Maps = [#ttcore.l1_map<{l1Bank = 0, dram = #ttcore.dram<{dramBank = 0, dramChan = 0}>, eth = #ttcore.eth<{ethChan = 0}>}>]}, physicalGrid = #ttcore.grid<8 x 10>, supportedDataTypes = [#ttcore.data_type<f32>, #ttcore.data_type<f16>, #ttcore.data_type<bf16>], supportedTileSizes = [#ttcore.tile_size<32 x 32>, #ttcore.tile_size<16 x 32>, #ttcore.tile_size<32 x 16>]}]}>

#loc = loc(unknown)
#loc1 = loc("ConvReLU":0:0)
#loc2 = loc("ConvReLU":1:0)
#loc3 = loc("ConvReLU":2:0)

func.func @forward(%arg0: tensor<1x3x32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"}, %arg1: tensor<16x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "conv_weight"}, %arg2: tensor<16xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "conv_bias"}) -> (tensor<1x16x30x30xbf16> {ttir.name = "output"}) {
    %0 = ttir.empty() : tensor<1x16x30x30xbf16> loc(#loc1)
    %1 = "ttir.conv2d"(%arg0, %arg1, %0) <{stride = [1 : i32, 1 : i32], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], dilation = [1 : i32, 1 : i32], groups = 1 : i32}> : (tensor<1x3x32x32xbf16>, tensor<16x3x3x3xbf16>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16> loc(#loc1)
    %2 = ttir.empty() : tensor<1x16x30x30xbf16> loc(#loc2)
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x16x30x30xbf16>, tensor<16xbf16>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16> loc(#loc2)
    %4 = ttir.empty() : tensor<1x16x30x30xbf16> loc(#loc3)
    %5 = "ttir.relu"(%3, %4) : (tensor<1x16x30x30xbf16>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16> loc(#loc3)
    return %5 : tensor<1x16x30x30xbf16> loc(#loc)
} 
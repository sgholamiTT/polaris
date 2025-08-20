#system_desc = #ttcore.system_desc<{chipIds = [0], chipDescs = [{arch = #ttcore.chip_arch<wormhole_b0>, grid = #ttcore.grid<8 x 10>, l1Maps = [#ttcore.l1_map<{l1Bank = 0, dram = #ttcore.dram<{dramBank = 0, dramChan = 0}>, eth = #ttcore.eth<{ethChan = 0}>}>]}, physicalGrid = #ttcore.grid<8 x 10>, supportedDataTypes = [#ttcore.data_type<f32>, #ttcore.data_type<f16>, #ttcore.data_type<bf16>], supportedTileSizes = [#ttcore.tile_size<32 x 32>, #ttcore.tile_size<16 x 32>, #ttcore.tile_size<32 x 16>]}]}>

#loc = loc(unknown)
#loc1 = loc("MLP":0:0)
#loc2 = loc("MLP":1:0)
#loc3 = loc("MLP":2:0)
#loc4 = loc("MLP":3:0)
#loc5 = loc("MLP":4:0)
#loc6 = loc("MLP":5:0)

func.func @forward(%arg0: tensor<1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"}, %arg1: tensor<128x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "fc1_weight"}, %arg2: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "fc1_bias"}, %arg3: tensor<64x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "fc2_weight"}, %arg4: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "fc2_bias"}) -> (tensor<1x32xbf16> {ttir.name = "output"}) {
    %0 = ttir.empty() : tensor<1x64xbf16> loc(#loc1)
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x128xbf16>, tensor<128x64xbf16>, tensor<1x64xbf16>) -> tensor<1x64xbf16> loc(#loc1)
    %2 = ttir.empty() : tensor<1x64xbf16> loc(#loc2)
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x64xbf16>, tensor<64xbf16>, tensor<1x64xbf16>) -> tensor<1x64xbf16> loc(#loc2)
    %4 = ttir.empty() : tensor<1x64xbf16> loc(#loc3)
    %5 = "ttir.relu"(%3, %4) : (tensor<1x64xbf16>, tensor<1x64xbf16>) -> tensor<1x64xbf16> loc(#loc3)
    %6 = ttir.empty() : tensor<1x32xbf16> loc(#loc4)
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<1x64xbf16>, tensor<64x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16> loc(#loc4)
    %8 = ttir.empty() : tensor<1x32xbf16> loc(#loc5)
    %9 = "ttir.add"(%7, %arg4, %8) : (tensor<1x32xbf16>, tensor<32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16> loc(#loc5)
    return %9 : tensor<1x32xbf16> loc(#loc)
} 
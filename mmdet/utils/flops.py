class FlopsCalculator(object):
    def __init__(self, layer=101, resolution=1000, density_dict=None):
        assert layer in [50, 101], 'Only Support ResNet 50/101 with Bottleneck Design.'

        # Model
        self.layer = layer
        if self.layer == 50:
            self.block = 'Bottleneck'
            self.blocks = [3, 4, 6, 3]
        elif self.layer == 101:
            self.block = 'Bottleneck'
            self.blocks = [3, 4, 23, 3]

        self.planes = [64, 128, 256, 512]
        self.strides = [4, 8, 16, 32]

        self.resolution = resolution
        if self.resolution == 224:
            # For Classification
            self.h = self.w = self.resolution
        else:
            # For Detection
            self.h = (self.resolution // 32) * 32
            self.w = (self.resolution * 5. / 3. // 32) * 32

        self.init_calculation()
        self.parse_density(density_dict=density_dict)

    def init_calculation(self):
        self.calculation = dict()

        self.calculation['conv1'] = self.conv_cal(7, 3, 64, 2)

        for idx_layer in range(4):
            self.calculation[f'layer{idx_layer + 1}'] = dict()
            if idx_layer == 0:
                _inplane = 16
            else:
                _inplane = self.planes[idx_layer - 1]
            _plane = self.planes[idx_layer]
            _stride = self.strides[idx_layer]

            self.calculation[f'layer{idx_layer + 1}']['downsample'] = self.conv_cal(1, _inplane * 4, _plane * 4, _stride)
            for idx_block in range(self.blocks[idx_layer]):
                if idx_layer == 0 and idx_block == 0:
                    _inplane = 16
                elif idx_layer != 0 and idx_block == 0:
                    _inplane = self.planes[idx_layer - 1]
                else:
                    _inplane = self.planes[idx_layer]
                _plane = self.planes[idx_layer]

                # First block in each layer but the first layer should have a different stride.
                if idx_layer != 0 and idx_block == 0:
                    _stride_1 = _stride / 2
                else:
                    _stride_1 = _stride

                self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}'] = dict()
                self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv1'] = self.conv_cal(1, _inplane * 4, _plane, _stride_1)
                self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv2'] = self.conv_cal(3, _plane, _plane, _stride)
                self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv3'] = self.conv_cal(1, _plane, _plane * 4, _stride)
                self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv1_mask'] = self.conv_cal(3, _inplane * 4, 1, _stride_1)
                self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv2_mask'] = self.conv_cal(3, _inplane * 4, 1, _stride)
                self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv1_interpolate'] = self.conv_cal(7, _plane, 1, _stride_1)
                self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv3_interpolate'] = self.conv_cal(7, _plane * 4, 1, _stride)
                if idx_layer != 0:
                    self.calculation[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv2_offset'] = self.conv_cal(3, _inplane * 4, 18, _stride)

    def get_whole_cal(self):
        c = self.calculation
        d = self.density
        def _inner(c, d):
            whole_cal = 0
            for key in c:
                if not isinstance(c[key], dict):
                    whole_cal += c[key] * d[key]
                else:
                    whole_cal += _inner(c[key], d[key])
            return whole_cal
        self.whole_cal = _inner(c, d)
        return self.whole_cal

    def parse_density(self, density_dict=None):
        self.density = dict()

        self.density['conv1'] = 1.
        for idx_layer in range(4):
            self.density[f'layer{idx_layer + 1}'] = dict()
            self.density[f'layer{idx_layer + 1}']['downsample'] = 1.
            for idx_block in range(self.blocks[idx_layer]):
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}'] = dict()
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv1'] = 1.
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv2'] = 1.
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv3'] = 1.
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv1_mask'] = 0.
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv2_mask'] = 0.
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv1_interpolate'] = 0.
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv3_interpolate'] = 0.
                self.density[f'layer{idx_layer + 1}'][f'block{idx_block}']['conv2_offset'] = 1.

        if density_dict is not None:
            for key_layer, layer_dict in density_dict.items():
                for key_block, block_dict in layer_dict.items():
                    self.density[key_layer][key_block]['conv1'] = block_dict['conv1']
                    self.density[key_layer][key_block]['conv2'] = block_dict['conv2']
                    self.density[key_layer][key_block]['conv3'] = block_dict['conv2']
                    self.density[key_layer][key_block]['conv1_mask'] = 1.
                    self.density[key_layer][key_block]['conv2_mask'] = 1.
                    self.density[key_layer][key_block]['conv1_interpolate'] = block_dict['conv1'] * (1 - block_dict['conv1'])
                    self.density[key_layer][key_block]['conv3_interpolate'] = block_dict['conv2'] * (1 - block_dict['conv2'])
                    self.density[key_layer][key_block]['conv2_offset'] = block_dict['conv2']

    def conv_cal(self, kernel, inplane, outplane, stride):
        return kernel * kernel * inplane * outplane * (self.h // stride) * (self.w // stride)

if __name__ == '__main__':
    for res in [500, 600, 800, 1000]:
        cc = FlopsCalculator(
            layer=101,
            resolution=res,
        )

        cal = cc.get_whole_cal()

        print(f'Baseline  : ResNet101\n'
              f'Resolution: {res}\n'
              f'Flops     : {cal / 1e9:.2f} GFlops\n')

import jax
from jax import numpy as jnp
from flax import linen as nn


class AxialDW(nn.Module):
    dim: int
    mixer_kernel: tuple=(7,7)
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        bn, h, w, c = x.shape
        dw_h = nn.Conv(self.dim, kernel_size=(h, 1), padding='SAME', feature_group_count=self.dim, kernel_dilation=self.dilation)
        dw_w = nn.Conv(self.dim, kernel_size=(1, w), padding='SAME', feature_group_count=self.dim, kernel_dilation=self.dilation)
        dw_h = dw_h(x)
        dw_w = dw_w(x)
        x = x + dw_h + dw_w
        return x


class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    in_c: int
    out_c: int
    mixer_kernel: tuple = (7,7)
    training: bool = True

    @nn.compact
    def __call__(self, x):
        skip = AxialDW(self.in_c, self.mixer_kernel)(x)
        skip = nn.BatchNorm(use_running_average=not self.training)(skip)

        x = nn.Conv(self.out_c, kernel_size=(1,1), padding='SAME')(skip)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))
        x = nn.gelu(x)
        return x, skip


class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    out_c: int
    mixer_kernel: tuple = (7,7)
    training: bool = True

    @nn.compact
    def __call__(self, x, skip):
        bn, h, w, c = x.shape
        x = jax.image.resize(x, shape=(bn, h*2, w*2, c), method='bilinear')
        x = jnp.concatenate([x, skip], axis=-1)
        x = nn.Conv(self.out_c, kernel_size=(1,1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not self.training)(x)
        x = AxialDW(self.out_c, mixer_kernel=self.mixer_kernel)(x)
        x = nn.Conv(self.out_c, kernel_size=(1,1), padding='SAME')(x)
        x = nn.gelu(x)
        return x


class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""
    dim: int
    training: bool = True

    @nn.compact
    def __call__(self, x):
        gc = self.dim // 4
        x = nn.Conv(gc, kernel_size=(1,1))(x)
        dw1 = AxialDW(gc, mixer_kernel=(3,3), dilation=1)(x)
        dw2 = AxialDW(gc, mixer_kernel=(3,3), dilation=2)(dw1)
        dw3 = AxialDW(gc, mixer_kernel=(3,3), dilation=3)(dw2)
        x = jnp.concatenate([x, dw1, dw2, dw3], axis=-1)
        x = nn.BatchNorm(use_running_average=not self.training)(x)
        x = nn.Conv(self.dim, kernel_size=(1,1), padding='SAME')(x)
        x = nn.gelu(x)
        return x


class ULite(nn.Module):
    features: int
    output_channels: int
    training: bool = True
    
    def setup(self):
        self.conv_in = nn.Conv(self.features, kernel_size=7, padding='SAME')
        self.e1 = EncoderBlock(self.features, self.features * 2, training=self.training)
        self.e2 = EncoderBlock(self.features * 2, self.features * 4, training=self.training)
        self.e3 = EncoderBlock(self.features * 4, self.features * 8, training=self.training)
        self.e4 = EncoderBlock(self.features * 8, self.features * 16, training=self.training)
        self.e5 = EncoderBlock(self.features * 16, self.features * 32, training=self.training)

        self.b5 = BottleNeckBlock(self.features * 32, training=self.training)

        self.d5 = DecoderBlock(self.features * 16, training=self.training)
        self.d4 = DecoderBlock(self.features * 8, training=self.training)
        self.d3 = DecoderBlock(self.features * 4, training=self.training)
        self.d2 = DecoderBlock(self.features * 2, training=self.training)
        self.d1 = DecoderBlock(self.features, training=self.training)

        self.conv_out = nn.Conv(1, kernel_size=1, padding='SAME')

    @nn.compact
    def __call__(self, x):
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        x = self.b5(x)

        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    # jax cpu
    jax.config.update("jax_platform_name", "cpu")
    key = jax.random.PRNGKey(0)

    model = ULite(16, 16, training=True)
    x = jnp.ones((2, 256, 256, 3))
    params = model.init(key, x)
    out, _ = model.apply(params, x, mutable=['batch_stats'])

    table_fn = nn.tabulate(
        model,
        key,
        compute_flops=True,
        compute_vjp_flops=True,
    )
    print(table_fn(x))

    for y in out:
        print(y.shape)

# coding: utf-8
# Author：WangTianRui
# Date ：2020/8/18 12:01

def get_net_params():
    params = {}
    encoder_dim_start = 32
    params["encoder_channels"] = [
        1,
        encoder_dim_start,     #32
        encoder_dim_start * 2, #64
        encoder_dim_start * 4, #128
        encoder_dim_start * 4, #128
        encoder_dim_start * 8, #256
        encoder_dim_start * 8  #256
    ]
    params["encoder_kernel_sizes"] = [
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2)
    ]
    params["encoder_strides"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    params["encoder_paddings"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    # ----------lstm---------

    params["lstm_dim"] = [
        1280, 128
    ]
    # this is for standard dccrn
    # in pvae, not 128. should be 384 = zdim(128) * 3
    params["dense"] = [
        128, 1280
    ] # params["lstm_dim"][1] and params["dense"][0] are not used in the pvae. they are controlled by zdim
    params["lstm_layer_num"] = 2
    # --------decoder--------
    params["decoder_channels"] = [
        encoder_dim_start * 8,
        encoder_dim_start * 8,
        encoder_dim_start * 4,
        encoder_dim_start * 4,
        encoder_dim_start * 2,
        encoder_dim_start * 1,
        1
    ]
    params["decoder_kernel_sizes"] = [
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2)
    ]
    params["decoder_strides"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    params["decoder_paddings"] = [
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0)
    ]
    # adapt according to input time frame num
    params["encoder_chw"] = [
        (32, 129, 1600),
        (64, 65, 1599),
        (128, 33, 1598),
        (128, 17, 1597),
        (256, 9, 1596),
        (256, 5, 1595)
    ]
    params["decoder_chw"] = [
        (256, 9, 1596),
        (128, 17, 1597),
        (128, 33, 1598),
        (64, 65, 1599),
        (32, 129, 1600),
        (1, 257, 1601)
    ]
    return params
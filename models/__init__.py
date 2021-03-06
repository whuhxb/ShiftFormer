from timm.models import *
from .poolformer import *
from .convnext import convnext_base, convnext_tiny
# from .shiftVIT import shiftvit_light_base
# from .model1_static_shift import model1_static_shiftformer_s12, model1_static_shiftformer_s12_n8, \
#     model1_static_shiftformer_s12_n16, model1_static_shiftformer_s24
# from .model2_static_shift import model2_static_shiftformer_s12
# from .model3_static_shift import model3_static_shiftformer_s12
# from .model4_static_shift import model4_static_shiftformer_s12
# from .model4_1_static_shift import model4_1_static_shiftformer_s12
# from .model5_static_shift import model5_static_shiftformer_s12
# from .model6_static_shift import model6_static_shiftformer_s12
# from .van import van_small
# from .model7_static_shift import model7_static_shiftformer_s12
# from .model8_static_shift import model8_static_shiftformer_s12
# from .model9_conv_former import model9_s12_3x3, model9_s12_5x5, model9_s12_3x3dilated2, model9_s12_7x7, \
#     model9_s12_3x3_7x7, model9_s12_9x9, model9_s12_3x3_7x7dilated2
# from .model10_static_shift import model10_static_shiftformer_s12, model10_static_shiftformer_s12_8844, model10_static_shiftformer_s24_8844
# from .model11_static_shift import model11_s12
# from .model12_static_shift import model12_s12
# from .model13_full_conv_transformer import fct_s12_64_7478_TTT, fct_s12_64_7478_TFT, fct_s12_64_7118_TTT, fct_s12_64_7478_TTF, \
#     fct_s24_64_7118_TTT_8844, fct_s12_64_7478_FFF, fct_s12_64_7418_TFT, fct_s12_64_7438_TTT
# from .model14_full_conv_transformer import fct14_s12_64_7478_FFF
# from .conv_pd_former import conv_pd_s12_pda, conv_pd_s12_pdb
# # from .van import van_tiny
#
# from .fct_shift_CDWshift import fct_s12_64_7118_TTT_CDWshift
# from .fct_shift_SDWshift import fct_s12_64_7118_TTT_SDWshift
# from .fct_shift_2DWshift import fct_s12_64_7118_TTT_2DWshift
# from .fct_channel_conv import   fct_s12_64_7118_TTT_ChannelConv
# from .fct_normpatch import      fct_s12_64_7118_TTT_normpatch
# from .fct_channel_att_eca import fct_s12_64_7118_TTT_eca
# from .fct_remove_cdwconv import fct_s12_64_7118_TTT_remove_cdwconv
# from .fct_lk_toekn import fct_s12_64_7118_TTT_lk_token
# from .fct_remove_sigmoid import fct_s12_64_7118_TTT_remove_sigmoid
#
# from .fcvt_v2_base import fcvt_s12_64_TTTT, fcvt_s12_64_FFFF, fcvt_s12_64_FTFF, fcvt_s12_64_TFTT, \
#     fcvt_s12_64_TFFF, fcvt_s12_64_FFFF_nogc
# from .fcvt_v3_base import fcvt_v3_s12_64_TFFF, fcvt_v3_s12_64_debug
# from .fcvt_v4_base import fcvt_v4_s12_64_TFFF_gc_weighted, fcvt_v4_s12_64_TFFF_gc_simple, fcvt_v4_s12_64_TFFF_nogc, \
#     fcvt_v4_s12_64_debug, fcvt_v4_s12_64_TTTT_gc_weighted

from .fcvt_v5_base import fcvt_v5_32_TTFF_W_3_3, fcvt_v5_32_TTFF_W_5_5, fcvt_v5_32_TTFF_W_5_9, fcvt_v5_32_TTFF_W_7_7, \
    fcvt_v5_32_TTFF_W_7_9, fcvt_v5_32_TTFF_W_9_9, fcvt_v5_32_TTFF_W_11_11, fcvt_v5_32_TTFF_W_13_13, \
    fcvt_v5_32_TFFF_11_11, fcvt_v5_32_TTFF_NotW_11_11,\
    fcvt_v5_32_TTFF_W_11_11_H1, fcvt_v5_32_TTFF_W_11_11_H4, fcvt_v5_32_TTFF_W_11_11_H16,fcvt_v5_32_TTFF_W_11_11_H32, \
    fcvt_v5_32_TTFF_W_11_11_H8_compete, \
    fcvt_v5_32_TTFF_W_11_11_H8_compete_TTF, fcvt_v5_32_TTFF_W_11_11_H8_compete_TTT,fcvt_v5_32_TTFF_W_11_11_H8_compete_FTT,fcvt_v5_32_TTFF_W_11_11_H8_compete_TFF,fcvt_v5_32_TTFF_W_11_11_H8_compete_TFT, \
    fcvt_v5_64_B12, fcvt_v5_64_B24, fcvt_v5_64_B48


# from .fcvt_v6_base import fcvt_v6_64_B12

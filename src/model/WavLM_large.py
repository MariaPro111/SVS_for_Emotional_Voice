# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wavlm/hubconf.py ]
#   Synopsis     [ the WavLM torch hubconf ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


import os

from src.model.wavlm_utils.download import _urls_to_filepaths

from src.model.wavlm_utils.expert import UpstreamExpert as _UpstreamExpert


def wavlm_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wavlm_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wavlm_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wavlm_large(refresh=False, *args, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt"

    return wavlm_url(refresh=refresh, *args, **kwargs)
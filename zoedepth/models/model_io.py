# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import os
from pathlib import Path
from urllib.parse import urlparse

def load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    # if model is a DataParallel model, then state_dict keys are prefixed with 'module.'

    do_prefix = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]

        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k

        state[k] = v

    model_state = model.state_dict()
    skipped_shape_keys = []
    compatible_state = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped_shape_keys.append(k)
            continue
        compatible_state[k] = v

    incompatible = model.load_state_dict(compatible_state, strict=False)
    if len(skipped_shape_keys) > 0:
        print(f"Skipped shape-mismatched keys: {len(skipped_shape_keys)}")
    if len(incompatible.unexpected_keys) > 0:
        print(f"Ignored unexpected keys: {len(incompatible.unexpected_keys)}")
    if len(incompatible.missing_keys) > 0:
        print(f"Missing keys: {len(incompatible.missing_keys)}")
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return load_state_dict(model, ckpt)


def load_state_dict_from_url(model, url, **kwargs):
    model_dir = kwargs.pop("model_dir", None)
    if model_dir is None:
        model_dir = str(Path(torch.hub.get_dir()) / "checkpoints")

    filename = Path(urlparse(url).path).name
    cached_file = str(Path(model_dir) / filename)

    try:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', model_dir=model_dir, **kwargs)
    except RuntimeError as e:
        msg = str(e)
        if ("PytorchStreamReader" in msg or "failed finding central directory" in msg) and Path(cached_file).exists():
            try:
                os.remove(cached_file)
            except OSError:
                pass
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', model_dir=model_dir, **kwargs)
        else:
            raise
    return load_state_dict(model, state_dict)


def load_state_from_resource(model, resource: str):
    """Loads weights to the model from a given resource. A resource can be of following types:
        1. URL. Prefixed with "url::"
                e.g. url::http(s)://url.resource.com/ckpt.pt

        2. Local path. Prefixed with "local::"
                e.g. local::/path/to/ckpt.pt


    Args:
        model (torch.nn.Module): Model
        resource (str): resource string

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    print(f"Using pretrained resource {resource}")

    if resource.startswith('url::'):
        url = resource.split('url::')[1]
        return load_state_dict_from_url(model, url, progress=True)

    elif resource.startswith('local::'):
        path = resource.split('local::')[1]
        return load_wts(model, path)
        
    else:
        raise ValueError("Invalid resource type, only url:: and local:: are supported")
    

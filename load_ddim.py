import sys
import os
import torch
from model_utils import *

def load_ddim(argv):
    from model_utils import load_model
    argv_ = sys.argv
    sys.argv = argv
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")
    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    gpu = True
    eval_mode = True
    print(config)
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    sampling_conf = vars(opt)
    print(sampling_conf)
    sys.argv = argv_
    return opt, model

import torch
import numpy as np
from lib.model.aggr.gpo import GPO
from collections import OrderedDict

import pickle


def main():
    gpool_model = GPO(32, 32)
    ckpt_file = './runs/coco_vsepp_updown_var_gpool/model_best.pth.tar'
    len_min = 5
    len_max = 50

    ckpt = torch.load(ckpt_file)
    state_dict = ckpt['model'][1]

    gpool_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if 'gpool' in name:
            idx = name.find('gpool')
            param_name = name[idx+6:]
            gpool_state_dict[param_name] = param

    gpool_model.load_state_dict(gpool_state_dict)

    lengths = np.arange(len_min, len_max)
    lengths = torch.Tensor(lengths)

    with torch.no_grad():
        weights, _ = gpool_model.compute_pool_weights(lengths)

        lengths = lengths.cpu().numpy()
        weights = weights.cpu().squeeze().numpy()

        weights_dict = {int(k): v for k, v in zip(lengths, weights)}

    out_path = './gpool_weights/gpool_grid_gru.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(weights_dict, f)



if __name__ == '__main__':
    main()

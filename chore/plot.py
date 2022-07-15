import os

import numpy as np
import matplotlib.pyplot as plt

from stefutil import *


if __name__ == '__main__':
    from ns_bi_encoder.util import *

    od_green = hex2rgb('#98C379', normalize=True)
    od_blue = hex2rgb('#619AEF', normalize=True)
    od_red = hex2rgb('#E06C75', normalize=True)
    od_purple = hex2rgb('#C678DD', normalize=True)

    hetero_data = {  # Context/Candidate
        'Siamese: BERT': dict(
            accs={'Banking77': 0.873051948051948, 'SNIPS': 0.966365461847389, 'Clinc_150': 0.864900662251655, 'SGD': 0.743394429897643},
            plt_kwargs=dict(c=od_green, marker='s'),
            num_param=14263680
        ),
        'Non-Siamese: BERT-BERT': dict(
            accs={'Banking77': 0.888311688311688, 'SNIPS': 0.975903614457831, 'Clinc_150': 0.888962472406181, 'SGD': 0.776719828612235},
            num_param=28527360,
            plt_kwargs=dict(c=od_blue, marker='8')
        ),
        'Non-Siamese: TinyBERT-BERT': dict(
            accs={'Banking77': 0.897727272727272, 'SNIPS': 0.983684738955823, 'Clinc_150': 0.895805739514348, 'SGD': 0.763389669126398},
            num_param=18649600,
            plt_kwargs=dict(c=od_red, marker='o')
        ),
        'Non-Siamese: BERT-TinyBERT': dict(
            accs={'Banking77': 0.887012987012987, 'SNIPS': 0.980923694779116, 'Clinc_150': 0.889183222958057, 'SGD': 0.733396810283265},
            num_param=18649600,
            plt_kwargs=dict(c=od_purple, marker='o'),

        )
    }

    """
    Plot heterogeneous performance, marker size reflective of model size   
    """
    # dnms = ['Banking77', 'SNIPS', 'Clinc_150', 'SGD']  # plot in that order
    dnms = ['SGD', 'Clinc_150', 'Banking77', 'SNIPS']
    n_dset = len(dnms)

    width = 4
    plt.figure(figsize=(width, width * 16/9))
    base_ms = 5  # dynamic size based on model #parameter
    ms_face_opacity = 0.5
    for model_name, d in hetero_data.items():
        accs = [d['accs'][dnm] * 100 for dnm in dnms]
        kws = d['plt_kwargs']
        if 'ms' not in kws:
            kws['ms'] = base_ms * d['num_param'] / 1e7
        plt.plot(accs, label=model_name, lw=0.75, **kws, markerfacecolor=[*kws['c'], ms_face_opacity])

    edge = 0.5
    ax = plt.gca()
    ax.set_xticks(list(range(n_dset)), labels=['Clinc-150' if d == 'Clinc_150' else d for d in dnms])
    ax.set_xlim([-edge, (n_dset-1) + edge])
    scores = np.concatenate([ln.get_ydata() for ln in ax.lines])
    ma, mi = np.max(scores), np.min(scores)
    ma, mi = min(round(ma, -1) + 10, 100), max(round(mi, -1), -5)
    ax.set_ylim([mi, ma])

    plt.legend()
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy (%)')

    save = True
    # save = False
    if save:
        fnm = f'{now(for_path=True)}, heterogeneous_performance.png'
        path = os.path.join(u.plot_path, fnm)
        plt.savefig(path, dpi=600)
    else:
        plt.show()

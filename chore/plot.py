import matplotlib.pyplot as plt

from stefutil import *


if __name__ == '__main__':
    hetero_data = {  # Context/Candidate
        'Siamese: BERT': {
            'Banking77': 0.873051948051948, 'SNIPS': 0.966365461847389, 'Clinc_150': 0.864900662251655, 'SGD': 0.743394429897643
        },
        'Non-Siamese: BERT-BERT': {
            'Banking77': 0.888311688311688, 'SNIPS': 0.975903614457831, 'Clinc_150': 0.888962472406181, 'SGD': 0.776719828612235
        },
        'Non-Siamese: TinyBERT-BERT': {
            'Banking77': 0.897727272727272, 'SNIPS': 0.983684738955823, 'Clinc_150': 0.895805739514348, 'SGD': 0.763389669126398
        },
        'Non-Siamese: BERT-TinyBERT': {
            'Banking77': 0.887012987012987, 'SNIPS': 0.980923694779116, 'Clinc_150': 0.889183222958057, 'SGD': 0.733396810283265
        }
    }

    """
    Plot heterogeneous performance, marker size reflective of model size   
    """
    dnms_order = ['Banking77', 'SNIPS', 'Clinc_150', 'SGD']
    plt.figure()
    for model_name, accs in hetero_data.items():
        accs = [accs[dnm] for dnm in dnms_order]
        plt.plot(accs, label=model_name)
    plt.legend()
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Heterogeneous performance')
    # plt.savefig('hetero_performance.png')
    plt.show()

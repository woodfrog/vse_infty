import os
import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    ckpt_bases = [
        #'runs/release_weights/coco_butd_region_bigru',
        #'runs/release_weights/coco_butd_grid_bigru',
        'runs/release_weights/f30k_butd_region_bigru',
        'runs/release_weights/f30k_butd_grid_bigru',
    ]

    for base in ckpt_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        #save_path = os.path.join(base, 'results_{}_5k.npy'.format(split))
        save_path = None

        # Evaluate COCO 5-fold 1K
        #evaluation.evalrank(model_path, data_path="/tmp/data/coco", split='testall', fold5=True, save_path=None)
        # Evaluate COCO 5K
        #evaluation.evalrank(model_path, data_path="/tmp/data/coco", split='testall', fold5=False, save_path=save_path)

        # Evaluate F30K
        evaluation.evalrank(model_path, data_path="/tmp/data/f30k", split='test', fold5=False, save_path=save_path)

        # Evlauate on CxC
        #evaluation.evalrank(model_path, data_path="/tmp/data/coco", split='testall', fold5=True, cxc=False)

    ## Evaluate the model ensemble
    #paths = ['runs/f30k_vsepp_updown_bert_var_gpool/results_testall_5k.npy',
    #         'runs/f30k_entity_bert_var_gpool/results_testall_5k.npy']

    #evaluation.eval_ensemble(results_paths=paths, fold5=True)
    #evaluation.eval_ensemble(results_paths=paths, fold5=False)

if __name__ == '__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main()

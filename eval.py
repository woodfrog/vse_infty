import os
import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    weights_bases = [
        # 'runs/coco_vsepp_updown_bert_var_gpool/',
        # 'data/weights/runs/coco_vsepp_wsl_bert_var_gpool/',
        # 'data/weights/runs/coco_vsepp_wsl_bert_var_gpool_2/',
        'runs/coco_entity_bert_var_gpool',
    ]
    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth.tar')
        # Save the final results for computing ensemble results
        # save_path = os.path.join(base, 'results_{}_5k.npy'.format(split))
        save_path = None

        # Evaluate COCO 5-fold 1K
        evaluation.evalrank(model_path, data_path="/tmp/data/coco", split='testall', fold5=True, save_path=None)
        # Evaluate COCO 5K
        evaluation.evalrank(model_path, data_path="/tmp/data/coco", split='testall', fold5=False, save_path=save_path)

        # Evaluate Flickr30K
        #evaluation.evalrank(model_path, data_path="data", split='test', fold5=False, save_path=save_path)

        # Evaluate on CxC 
        #evaluation.evalrank(model_path, data_path="data/coco", split='testall', fold5=True, cxc=False)

    # Evaluate model ensemble
    #paths = ['runs/f30k_vsepp_updown_bert_var_gpool/results_testall_5k.npy',
    #         'runs/f30k_entity_bert_var_gpool/results_testall_5k.npy']

    #evaluation.eval_ensemble(results_paths=paths, fold5=True)
    #evaluation.eval_ensemble(results_paths=paths, fold5=False)

if __name__ == '__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main()

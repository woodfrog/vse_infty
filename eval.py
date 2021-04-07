import os
import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():

    split = 'testall'
    bases = [
        'data/weights/runs/coco_vsepp_updown_bert_var_gpool/',
        # 'data/weights/runs/coco_vsepp_wsl_bert_var_gpool/',
        # 'data/weights/runs/coco_vsepp_wsl_bert_var_gpool_2/',
        # 'data/weights/runs/coco_entity_bert_var_gpool',
    ]
    for base in bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth.tar')
        #save_path = os.path.join(base, 'results_{}_5k.npy'.format(split))
        save_path = None

        #evaluation.evalrank(model_path, data_path="data", split='test', fold5=False, save_path=save_path)
        evaluation.evalrank(model_path, data_path="data/coco", split='testall', fold5=True, cxc=False)
        #evaluation.evalrank(model_path, data_path="data", split='testall', fold5=True, save_path=None)
        #evaluation.evalrank(model_path, data_path="data", split='testall', fold5=False, save_path=save_path)

    #paths = ['runs/f30k_vsepp_updown_bert_var_gpool/results_testall_5k.npy',
    #         'runs/f30k_entity_bert_var_gpool/results_testall_5k.npy']

    #evaluation.eval_ensemble(results_paths=paths, fold5=True)
    #evaluation.eval_ensemble(results_paths=paths, fold5=False)

if __name__ == '__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main()

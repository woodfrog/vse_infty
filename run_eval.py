import os
import evaluation

import logging


def main():
    split = 'testall'
    bases = ['runs/coco_vsepp_updown_bert_var_gpool_test/', ]
    for base in bases:
        #logging_path = os.path.join(base, 'test_log_release.txt')
        #logging.basicConfig(filename=logging_path, filemode='w', format='%(asctime)s %(message)s', level=logging.INFO)
        model_path = os.path.join(base, 'model_best.pth.tar')
        #save_path = os.path.join(base, 'results_{}_5k.npy'.format(split))
        save_path = None
    
        #evaluation.evalrank(model_path, data_path="data", split='test', fold5=False, save_path=save_path)
        evaluation.evalrank(model_path, data_path="data", split='testall', cxc=True)
        #evaluation.evalrank(model_path, data_path="data", split='testall', fold5=True, save_path=None)
        #evaluation.evalrank(model_path, data_path="data", split='testall', fold5=False, save_path=save_path)
    
    #paths = ['runs/f30k_vsepp_updown_bert_var_gpool/results_testall_5k.npy',
    #         'runs/f30k_entity_bert_var_gpool/results_testall_5k.npy']
    
    #evaluation.eval_ensemble(results_paths=paths, fold5=True)
    #evaluation.eval_ensemble(results_paths=paths, fold5=False)

if __name__ == '__main__':
    main()

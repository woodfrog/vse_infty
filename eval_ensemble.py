from lib import evaluation

# Evaluate model ensemble
paths = ['runs/coco_butd_grid_bert/results_coco.npy',
         'runs/coco_butd_region_bert/results_coco.npy']

evaluation.eval_ensemble(results_paths=paths, fold5=True)
evaluation.eval_ensemble(results_paths=paths, fold5=False)
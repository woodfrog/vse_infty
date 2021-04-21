import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Evaluate model ensemble
paths = ['runs/release_weights/coco_butd_grid_bigru/results_coco.npy',
         'runs/release_weights/coco_butd_region_bigru/results_coco.npy']

evaluation.eval_ensemble(results_paths=paths, fold5=True)
evaluation.eval_ensemble(results_paths=paths, fold5=False)

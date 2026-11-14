"""Read-only first-round validation diagnostics; no test scores read."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.compare_thresholds import select_threshold
from src.dataset import LesionDataset, MEAN, STD


def main():
    root=Path('outputs/round2')
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dataset=LesionDataset('data/splits/val.csv')
    index={name:i for i,name in enumerate(dataset.frame.image_name)}
    frame=pd.read_csv('outputs/experiments/bce_seed42/best_validation_predictions.csv',float_precision='round_trip')
    threshold=select_threshold(frame.target,frame.probability,.9)
    missed=frame[(frame.target==1)&(frame.probability<threshold)]
    missed.to_csv(root/'baseline_validation_misses.csv',index=False)
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for ax,row in zip(axes.flat,missed.itertuples()):
        image,_,_=dataset[index[row.image_name]]
        rgb=(image.numpy().transpose(1,2,0)*np.asarray(STD)+np.asarray(MEAN)).clip(0,1)
        ax.imshow(rgb)
        ax.set_title(f'{row.image_name}\nscore={row.probability:.5f}',fontsize=8)
        ax.axis('off')
    fig.suptitle('BCE validation false negatives at sensitivity-target threshold; model inputs')
    fig.tight_layout()
    fig.savefig(root/'baseline_validation_misses.png',dpi=160)
    plt.close(fig)
    curves={}
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,name in zip(axes,['bce_seed42','weighted_bce_seed42','focal_seed42']):
        h=pd.read_csv(f'outputs/experiments/{name}/history.csv')
        ax.plot(h.epoch,h.train_loss,label='Train')
        ax.plot(h.epoch,h.val_loss,label='Validation')
        ax.set(title=name,xlabel='Epoch',ylabel='Loss')
        ax.legend()
        curves[name]=dict(best_auroc=float(h.val_auroc.max()),last_auroc=float(h.val_auroc.iloc[-1]))
    fig.tight_layout()
    fig.savefig(root/'round1_loss_diagnostics.png',dpi=160)
    plt.close(fig)
    (root/'diagnostics.json').write_text(json.dumps(dict(validation_only=True,curves=curves,
        hypotheses=['Original best checkpoints occur early; more epochs alone have not improved validation AUROC.',
                    'Batch size 8 and rare positives motivate testing frozen BatchNorm statistics and moderate positive resampling.',
                    'These are hypotheses, not proven diagnoses or guarantees of improvement.'],
        baseline_validation_false_negatives=len(missed)),indent=2))
    print('Saved validation-only diagnostic figures and hypotheses')


if __name__=='__main__':
    main()

Userful links:

- kaggle competition: [kaggle.com](https://www.kaggle.com/c/contradictory-my-dear-watson/data)
- learning bert for
  pytorch: [kaggle.com](https://www.kaggle.com/vbookshelf/basics-of-bert-and-xlm-roberta-pytorch/notebook)

Start training on kaggle machine:

```bash
! git clone https://github.com/homomorfism/text-classification-bert.git || (cd text-classification-bert ; git pull)
! pip install -qqq -r text-classification-bert/requirements.txt
! cd text-classification-bert && python train.py
```

Need to implement:

- cross validation with 5-6 folds
- add lr schedulers
- augmentations (e.x. [augmentation with translation]())

Observation logs from experiments:

- Bert's one epoch = 5.3 minute - we could have around 20 epochs for 1 experiment

Training logs:

- 18.01.2022 - 9 epochs, final accuracy 0.64485
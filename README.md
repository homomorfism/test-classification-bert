Userful links:

- kaggle competition: [kaggle.com](https://www.kaggle.com/c/contradictory-my-dear-watson/data)
- learning bert for
  pytorch: [kaggle.com](https://www.kaggle.com/vbookshelf/basics-of-bert-and-xlm-roberta-pytorch/notebook)

Start training on kaggle machine:

```bash
! if cd text-classification-bert; then git pull; else https://github.com/homomorfism/text-classification-bert.git; fi
! pip install -qqq -r text-classification-bert/requirements.txt
! cd text-classification-bert && python train.py
```
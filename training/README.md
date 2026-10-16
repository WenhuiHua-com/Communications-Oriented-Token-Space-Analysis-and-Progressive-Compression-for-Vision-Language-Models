# Codebook training

Run from the repository root using the BLIP environment and pretrained assets described in the main README.

Before running, edit `New_code-book-search-fast.py`:

- Set `train_image_root`, `val_image_root` and the pretrained-model paths. Check the BERT path in `models/blip.py`.
- Set the ImageNet sample count in `random.sample` and its preceding condition to match your dataset size.
- Set `image_size` and the model image sizes consistently, including `model_retrieval` used for evaluation.
- Set `codebook_size` and `sparsity_percentage` as needed.

```bash
PYTHONPATH=. python training/New_code-book-search-fast.py
```

The script runs sequentially: it saves visual tokens to `image_token_0.5.pth` and importance scores to `importance_score_0.5.pth`, reads these caches to train the codebook, saves `codebook_retrieval_New_复现.pth`, and then evaluates it. Output files are written to the current working directory.

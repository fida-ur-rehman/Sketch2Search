# Sketch2Search: Evaluating CLIP Embeddings for Sketch-Based Image Retrieval

Sketch2Search explores whether **CLIP ViT-B/32** can retrieve relevant
photographs from hand-drawn sketches, and whether fine-tuning improves
retrieval ranking.

## Project overview

Sketches and photographs represent objects differently: sketches often
contain simple outlines, while photographs include colour, texture,
lighting, and background details.

**Research question:** How effectively can pretrained CLIP retrieve
relevant photographs from hand-drawn sketches, and does fine-tuning
improve performance?

## Dataset

The experiment uses a subset of the Sketchy database:

-   300 sketches and 300 photographs across 15 categories
-   Training: 209 pairs representing 37 objects
-   Validation: 41 pairs representing 8 objects
-   Test: 50 sketches representing 9 independent objects
-   Strict retrieval gallery: 255 photographs

Images were extracted by scanning the archive sequentially, which may
bias the sample toward items appearing earlier. This subset should not
be treated as representative of the complete database.

The data was accessed through the `DrRORAL/sketchy-dataset` repository
on Hugging Face. Check the original dataset licence and terms before
redistributing images.

## Method

1.  Load `openai/clip-vit-base-patch32` and its processor.
2.  Compare the pretrained baseline with a fine-tuned model.
3.  Fine-tune for five epochs using AdamW, learning rate `1e-6`, weight
    decay `0.01`, and a multi-positive contrastive loss.
4.  Split data by object ID to keep training, validation, and test
    objects separate.
5.  Extract and L2-normalise embeddings for sketches and gallery
    photographs.
6.  Rank photographs using cosine similarity.
7.  Evaluate Recall@1, Recall@5, Recall@10, and Mean Reciprocal Rank
    (MRR), averaging sketch scores within each object before averaging
    across the nine test objects.
8.  Select the checkpoint with the lowest validation loss.

## Results

### Retrieval performance

  Metric        Pretrained CLIP   Fine-tuned CLIP
  ----------- ----------------- -----------------
  Recall@1               0.0111            0.0185
  Recall@5               0.1778            0.2407
  Recall@10              0.3370            0.4873
  MRR                    0.1261            0.1710

Fine-tuning improved all four metrics. The largest absolute gain was in
Recall@10, suggesting that relevant photographs were more likely to
appear in the top-ten shortlist. Recall@1 remained very low, so the
first result alone was unreliable.

### Training and validation loss

    Epoch   Training loss   Validation loss
  ------- --------------- -----------------
        1          1.0464            1.3993
        2          0.5219            1.4050
        3          0.3057            1.4388
        4          0.2642            1.4927
        5          0.2144            1.5325

Training loss decreased while validation loss increased, a pattern
consistent with possible overfitting. The best validation loss was
`1.3993` at epoch 1.

## Running the project

The experiment was developed in Google Colab.

1.  Open the notebook in Google Colab.
2.  Enable a GPU runtime if available.
3.  Install or import the notebook's dependencies, including PyTorch,
    Transformers, NumPy, pandas, scikit-learn, and Pillow.
4.  Follow the notebook's data-loading steps to obtain the Sketchy
    subset.
5.  Run the cells in order to prepare the data, train the model, select
    the checkpoint, create the strict gallery, and evaluate both models.

Exact package versions and a standalone command-line entry point are not
specified here. Add them if you want to reproduce the project outside
Colab.

## Suggested repository structure

``` text
Sketch2Search/
├── README.md
├── notebooks/
│   └── Sketch2Search.ipynb
├── figures/
│   ├── training_validation_loss.png
│   ├── retrieval_metrics.png
│   └── qualitative_retrieval_examples.png
└── requirements.txt
```

This is a suggested structure, not a claim about files already in the
repository. Update filenames to match your actual files.

## Limitations and future work

-   Only nine independent test objects were evaluated.
-   Sequential archive extraction may have introduced sampling bias.
-   Increasing validation loss suggests possible overfitting.
-   Only one training run was evaluated.
-   The results are an initial experiment, not evidence of readiness for
    general real-world use.

Future work should evaluate more objects and categories, use
reproducible sampling, repeat runs with different random seeds, estimate
uncertainty, and compare frozen-encoder or partial fine-tuning
strategies.

## References

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S.,
Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., &
Sutskever, I. (2021). *Learning transferable visual models from natural
language supervision*. Proceedings of the 38th International Conference
on Machine Learning, 139, 8748--8763.
https://proceedings.mlr.press/v139/radford21a.html

Sangkloy, P., Burnell, N., Ham, C., & Hays, J. (2016). *The Sketchy
database: Learning to retrieve badly drawn bunnies*. ACM Transactions on
Graphics, 35(4), Article 119. https://doi.org/10.1145/2897824.2925954

## Author

**Fida Ur Rehman**

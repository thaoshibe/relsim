# relsim

<p align="center">
  <a href="PUT_ARXIV_LINK_HERE"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-blue" alt="BibTeX"></a>
  <a href="https://thaoshibe.github.io/relsim/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/thaoshibe/anonymous-captions-114k"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="HuggingFace Dataset"></a>
</p>

| ![](https://thaoshibe.github.io/relsim/images/peach-earth.png) |
|:--:|
| *We introduce a new visual similarity notion — relational visual similarity (relsim) — which captures the internal relational logic of a scene rather than its surface appearance.* |

[**Relational Visual Similarity**](https://thaoshibe.github.io/relsim/) (arXiv 2025)  
[Thao Nguyen](https://thaoshibe.github.io/)<sup>1</sup>, [Sicheng Mo](https://sichengmo.github.io/)<sup>3</sup>, [Krishna Kumar Singh](https://krsingh.cs.ucdavis.edu/)<sup>2</sup>, [Yilin Wang](https://yilinwang.org/)<sup>2</sup>, [Jing Shi](https://jshi31.github.io/jingshi/)<sup>2</sup>, [Nicholas Kolkin](https://scholar.google.com/citations?user=MqWYTj0AAAAJ&hl=en)<sup>2</sup>, [Eli Shechtman](https://scholar.google.com/citations?user=B_FTboQAAAAJ&hl=en)<sup>2</sup>, [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)<sup>1,2, ★</sup>, [Yuheng Li](https://yuheng-li.github.io/)<sup>1, ★</sup>

(★ Equal advising)

1- University of Wisconsin–Madison | 2- Adobe Research | 3- UCLA

> TL;DR: We introduce a new visual similarity notion — **relational visual similarity** — which complements traditional **attribute-based perceptual similarity** (e.g., LPIPS, CLIP, DINO).

---

🔗 Jump to: [Requirements](#requirements) | [Usage](#usage) | [Data](#data) | [BibTeX](#bibtex) |

# Data <a name="data"></a>

> You can see the snapshot of the data here: [relsim: data viewer](https://thaoshibe.github.io/relsim/data_viewer/index.html)

create a table here: dataset name, why is in side

| Dataset name | Short description  | JSON file | 🔍 Data viewer |
|--------------|-----------------|------------|------------|
| seed-groups <a href="https://huggingface.co/datasets/thaoshibe/anonymous-captions-114k"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="HuggingFace Dataset"></a> | Use to train the anonymous captioning model | [seed_groups.json](./data/seed_groups.json) | [See Seed Groups Dataset](https://thaoshibe.github.io/relsim/data_viewer/seed_groups.html) |
| anonymous-captions-114k <a href="https://huggingface.co/datasets/thaoshibe/anonymous-captions-114k"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="HuggingFace Dataset"></a> | Use to train the relational similarity model | [anonymous_captions_train.jsonl](./data/anonymous_captions_train.jsonl), [anonymous_captions_test.jsonl](./data/anonymous_captions_test.jsonl)| [See Anonymous Captions Dataset](https://thaoshibe.github.io/relsim/data_viewer/anonymous_captions.html) |

Each image will be given by their corresponding Image URL. Please see the corresponding json file in [./data](./data); or see live [data-viewer](https://thaoshibe.github.io/relsim/data_viewer/index.html).

To download, please run this the [data/download_data.py](./data/download_data.sh)

```
#--- Clone the repo if you haven't do that
# git clone https://github.com/thaoshibe/relsim.git
# cd relsim

#--- To download all dataset
bash data/download_data.sh

# (Optional) Depending on your internet speed, it should take under 0.5 hours to download all images with the default MAX_WORKER = 64.
# You can increase MAX_WORKER to speed up the download or reduce it depending on your machine (see the data/download_data.sh)
# For example:

python data/download_data.py --json_file data/anonymous_captions_test.jsonl \
    --save_dir data/anonymous_captions_test_images \
    --max_workers 128
```

## Disclaimer
> All images are extracted from [LAION](https://laion.ai/) dataset. We do **NOT** own any of the images and we acknowledge the rights and contributions of the original creators. Please respect the authors of all images. These images are used for **research purposes only**.

---

## BibTeX <a name="citation"></a>

```bibtex
@article{nguyen2025relsim,
  title={Relational Visual Similarity},
  author={Nguyen, Thao and Mo, Sicheng and Singh, Krishna Kumar and Wang, Yilin and Shi, Jing and Kolkin, Nicholas and Shechtman, Eli and Lee, Yong Jae and Li, Yuheng},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}

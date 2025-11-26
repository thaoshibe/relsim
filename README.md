# relsim

<p align="left">
  <a href="PUT_ARXIV_LINK_HERE"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-blue" alt="BibTeX"></a>
  <a href="https://thaoshibe.github.io/relsim/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/thaoshibe/anonymous-captions-114k"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="HuggingFace Dataset"></a>
  <a href="https://thaoshibe.github.io/relsim/data_viewer/index.html"><img src="https://img.shields.io/badge/Data-Viewer-green" alt="Data Viewer"></a>
</p>

| ![](https://thaoshibe.github.io/relsim/images/peach-earth.png) |
|:--:|
| *We introduce a new visual similarity notion — relational visual similarity (relsim) — which captures the internal relational logic of a scene rather than its surface appearance.* |

[**Relational Visual Similarity**](https://thaoshibe.github.io/relsim/) (arXiv 2025)  
[Thao Nguyen](https://thaoshibe.github.io/)<sup>1</sup>, [Sicheng Mo](https://sichengmo.github.io/)<sup>3</sup>, [Krishna Kumar Singh](https://krsingh.cs.ucdavis.edu/)<sup>2</sup>, [Yilin Wang](https://yilinwang.org/)<sup>2</sup>, [Jing Shi](https://jshi31.github.io/jingshi/)<sup>2</sup>, [Nicholas Kolkin](https://scholar.google.com/citations?user=MqWYTj0AAAAJ&hl=en)<sup>2</sup>, [Eli Shechtman](https://scholar.google.com/citations?user=B_FTboQAAAAJ&hl=en)<sup>2</sup>, [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)<sup>1,2, ★</sup>, [Yuheng Li](https://yuheng-li.github.io/)<sup>1, ★</sup>
<br>(★ Equal advising)
<br> 1- University of Wisconsin–Madison | 2- Adobe Research | 3- UCLA

> TL;DR: We introduce a new visual similarity notion: **relational visual similarity**, which complements traditional **attribute-based perceptual similarity** (e.g., LPIPS, CLIP, DINO).

---

🔗 Jump to: [Requirements](#requirements) | [Usage](#usage) | [🫥 Anonymous Captioning Model](#anonymousmodel) | [📁 Data](#data) | [BibTeX](#citation) |

# Usage <a name="anonymousmodel"></a>

Given two images, you can compute their relational visual similarity like this:


# 🫥 Anonymous Caption Model <a name="anonymousmodel"></a>

The pretrained anonymous caption model (Qwen-VL-2.5 7B) is provided in [./anonymous_caption](./anonymous_caption/).  
This model is trained on a limited number of seed groups and their corresponding generated captions (you can see the training data [here](https://thaoshibe.github.io/relsim/data_viewer/seed_groups.html)).

```python
python anonymous_caption/anonymous_caption.py # this will run on default test image
python anonymous_caption/anonymous_caption.py --image_path $PATH_TO_IMAGE_OR_IMAGE_FOLDER # run on your own images

python anonymous_caption/anonymous_caption.py --help # if you need to see all arguments
```

Here is example of the generated captions with different runs.
| Input Image | Generated Captions (Different run) |
|-----|-------|
| <img src="./anonymous_caption/mam.jpg" height="150"> | `python anonymous_caption/anonymous_caption.py --image_path anonymous_caption/mam.jpg`<br>Run 1: "Curious {Animal} peering out from behind a {Object}."<br> Run 2: "Curious {Animal} peeking out from behind the {Object} in an unexpected and playful way."<br> Run 3: "Curious {Cat} looking through a {Doorway} into the {Room}."<br> Run 4: "A curious {Animal} peeking from behind a {Barrier}."<br> Run 5: "A {Cat} peeking out from behind a {Door} with curious eyes."<br>... |
| <img src='./anonymous_caption/bo.jpg' height="150"> | `python anonymous_caption/anonymous_caption.py --image_path anonymous_caption/bo.jpg`<br>Run 1: "Animals with {Leaf} artfully placed on their {Head}."<br> Run 2: "A {Dog} with a {Leaf} delicately placed on its head."<br> Run 3: "A {Dog} with a {Leaf} artfully placed on its head."<br> Run 4: "A {Dog} with a {Leaf} delicately placed on their head, representing the beauty of {Season}."<br> Run5: "Animals adorned with {Leaf} in a {Seasonal} setting."<br> ...| 

> ⚠️ The anonymous caption can definerly can be improve... Open challengges [THAO FILL IN HERE]

# 📁 Data <a name="data"></a>

> **🔍 You can see the snapshot of the data here: [🔍🔍🔍 relsim: data viewer](https://thaoshibe.github.io/relsim/data_viewer/index.html)**

| Dataset name | Short description  | JSON file | 🔍 Data viewer |
|--------------|-----------------|------------|------------|
| seed-groups <a href="https://huggingface.co/datasets/thaoshibe/seed-groups"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="HuggingFace Dataset"></a> | Use to train the anonymous captioning model | [seed_group.json](./data/seed_group.json) | [See Seed Groups Dataset](https://thaoshibe.github.io/relsim/data_viewer/seed_groups.html) |
| anonymous-captions-114k <a href="https://huggingface.co/datasets/thaoshibe/anonymous-captions-114k"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="HuggingFace Dataset"></a> | Use to train the relational similarity model | [anonymous_captions_train.jsonl](./data/anonymous_captions_train.jsonl), [anonymous_captions_test.jsonl](./data/anonymous_captions_test.jsonl)| [See Anonymous Captions Dataset](https://thaoshibe.github.io/relsim/data_viewer/anonymous_captions.html) |

Each image will be given by their corresponding Image URL. Please see the json files in [./data](./data).
<br>(Optional) Depending on your internet speed, it should take under 0.5 hours to download all images with the default MAX_WORKER = 64.
You can increase MAX_WORKER to speed up the download or reduce it depending on your machine (see the [data/download_data.sh](./data/download_data.sh))

To download, please run this the [data/download_data.sh](./data/download_data.sh)

```
# download dataset

git clone https://github.com/thaoshibe/relsim.git #clone the repo if you haven't do that
cd relsim

bash data/download_data.sh # this script will download all dataset
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

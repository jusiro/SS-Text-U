
# Semi-Supervised Few-Shot Adaptation of VLMs

<img src="./documents/media/overview.svg" width = "550" alt="" align=center /> <br/>

[*Semi-Supervised Few-Shot Adaptation of Vision-Language Models*]().<br/>
[******](******), [******](******),
[******](******), [******](******), ⋅ ******.
<br/>


### Install

* Install in your environment a compatible torch version with your GPU. For example:

```
conda create -n sstextu python=3.11 -y
conda activate sstextu
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

```
git clone https://github.com/******/SS-Text-U.git
cd SS-Text-U
pip install -r requirements.txt
```

### Preparing the datasets
- Configure data paths (see [`./documents/local_data/constants.py`](./documents/local_data/constants.py)).
- Download, and configure datasets (see [`./documents/local_data/datasets/README.md`](./documents/local_data/datasets/README.md)).

## Usage
We present the basic usage here.

(a) Features extraction:
- `python extract_features.py`

(b) Standard few-shot adaptation:
- `python adapt.py --adapt SS-Text+`

(c) Semi-supervised few-shot adaptation::
- `python adapt.py --adapt SS-Text-U`

You will find the results upon training at [`./documents/local_data/results/`](./documents/local_data/results/).

## Citation

If you find this repository useful, please consider citing the following sources.

```
@inproceedings{sstextu,
    title={Semi-Supervised Few-Shot Adaptation of Vision-Language Models},
    author={******},
    booktitle={arXiv preprint arXiv:xxxx.xxxxx},
    year={2026}
}
```




















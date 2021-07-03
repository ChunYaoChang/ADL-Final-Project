# Description

## Requirements

```bash
pip install -r utils/requirements.txt
```

## Data

Please refer to the `README.md` in folder `data`.

## Preprocess

First put the data into `data`. Then, execute the following command.

```bash
python preprocess.py
```

## Download Checkpoint

Run the folllowing command to get the checkpoint.

```bash
wget https://www.dropbox.com/s/1l3pkrak526anqz/28675_05243.ckpt?dl=1 -O 28675_05243.ckpt
```

## Training

Go to `config.py` to adjust the parameters.

```bash
python T5.py
```

## Testing

Go to `config.py` to adjust the parameters.

```bash
# test_seen
python T5.py --mode test --load_from_checkpoint 28675_05243.ckpt --test_type seen --output_path ./seen.json
python postprocess.py ./seen.json ./result_seen.json

# test_unseen
python T5.py --mode test --load_from_checkpoint 28675_05243.ckpt --test_type unseen --output_path ./unseen.json
python postprocess.py ./unseen.json ./result_unseen.json
```

## Citations

This task take the following papar as reference.

```
@inproceedings{lin2021leveraging,
  title={Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue StateTracking},
  author={Lin, Zhaojiang and Liu, Bing and Moon, Seungwhan and Crook, Paul A and Zhou, Zhenpeng and Wang, Zhiguang and Yu, Zhou and Madotto, Andrea and Cho, Eunjoon and Subba, Rajen},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={5640--5648},
  year={2021}
}
```

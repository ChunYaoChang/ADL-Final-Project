## Descriptions
Here is the first method we applied in NLG task. We use pretrained Blenderbot to generate additional chit-chat in this method.
## Dependencies
```
torch 1.8.0
transformers 4.6.0
```
## Data
Please refer to the README.md in folder data.
## Training Classifiers
#### Beginning Chit-Chat Classifier
```
python3 train.py ./data/ --train_type beginning
```
#### End Chit-Chat Classifier
```
python3 train.py ./data/ --train_type end
```
## Download Checkpoint
```
wget https://www.dropbox.com/s/tadch78klo0ggsq/BeginClassifier.mdl?dl=1 -O BeginClassifier.mdl
wget https://www.dropbox.com/s/v3sj5neu616oxq1/EndClassifier.mdl?dl=1 -O EndClassifier.mdl
```
## Inference
#### Format 
```
python3 inference.py ./data/[test data dir] [Beginning classifier model path] [End classifier model path] [Result Path]
```
#### Example
```
python3 inference.py ./data/test_seen/ ./BeginClassifier.mdl ./EndClassifier.mdl ./result.json
```
## Citation
```
@inproceedings{lin2021leveraging,
  title={Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue StateTracking},
  author={Lin, Zhaojiang and Liu, Bing and Moon, Seungwhan and Crook, Paul A and Zhou, Zhenpeng and Wang, Zhiguang and Yu, Zhou and Madotto, Andrea and Cho, Eunjoon and Subba, Rajen},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={5640--5648},
  year={2021}
}
```

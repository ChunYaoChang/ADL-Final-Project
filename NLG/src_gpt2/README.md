## Descriptions
Here is the second method we applied in NLG task. We fine-tuned a pretrained GPT-2 model to generate additional chit-chat in this method.
## Dependencies
```
python==3.9.5
numpy>=1.20
torch==1.8.1
transformers==4.6.0
accelerate==0.3.0
```
## Fine-tune
```
python3.9 train.py
```
## Download Checkpoint
```
bash download.sh
```
## Inference
#### Format 
```
python3.9 generate_seen.py [output path]
```
Default output path is ./output.json.

#### Example
```
python3.9 generate_seen.py ./result.json

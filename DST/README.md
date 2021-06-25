# Description

## Requirements

```bash
pip install -r utils/requirements.txt
```

## Preprocess

First put the data into `data`. Then, execute the following command.

```bash
python preprocess.py
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
python T5.py --mode test --test_type seen --output_path ./seen.json
python postprocess.py ./seen.json ./result_seen.json

# test_unseen
python T5.py --mode test --test_type unseen --output_path ./unseen.json
python postprocess.py ./unseen.json ./result_unseen.json
```
# Vector Quantized Intent Contrastive Learning for Sequential Recommendation (VQ-ICL)

The source code is used for review in ICASSP 2026.


## Implementation
### Requirements
```
python>=3.9
Pytorch>=1.12.0
torchvision==0.13.0
torchaudio==0.12.0
numpy==1.24.4
scipy==1.6.0
pandas==2.2.3
```
### Datasets
Three public datasets are included in the `datasets` folder. (Beauty, Toys, ML-1M)

### Train VQ-ICL
To train VQ-ICL with three datasets, you can run the following command: 
```
bash scripts/train_all.sh
```
The script will automatically train VQ-ICL, save the best model based on the validation set, and then evaluate it on the test set.

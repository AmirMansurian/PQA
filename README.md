# PQA
Persian QesutionAnswering

Contributors : Amir Mansourian - Elnaz Rahmati - Zeinab Taghavi - Sepehr Ghobadi - Vida Ramezanian

Documentation: https://docs.google.com/document/d/1GRqZdRClQduQxV5ld0vCRSdLCTcN85yAr-RdA8X2tGw/edit?usp=sharing

Trained models: (put the links to trained models here)
<br/>
**ParsBERT**: [ParsBERT Model](https://drive.google.com/drive/folders/10j3KSd0zu4eM94yNKH7B-4grd5PhkeT_?usp=sharing)
<br/>
**mBERT**: [mBERT Model](https://drive.google.com/drive/folders/1Pk4U5XfXuT0zgLPCnxdZN-h4fDA_qzro?usp=sharing)
<br/>
XLM-RoBERTa:
<br/>
**ParsT5**: The model was too big to finetune on colab, so for just one epoch, I had to split training data to 4 batches and train the model in four steps. accuracy after one epoch was 0.462 and model didn't converge so better results can be achieved if we had access to more computational resources. You can find a tutorial on how to use this model in this path: Models/ParsT5_test.ipynb and finally link to model: https://drive.google.com/drive/folders/1iKHu4Wr8_5MNysVfBd8PhROzANSQyBOm?usp=sharing



### Results

|   Model  |  EM  | F1 | BLEU | BLEU1 | BLEU4 | 
|:----------:|:---------:|:------------:|:------------:|:------------:|:------------:|
| mBERT | **0.6484** |    **0.6926**   | **0.4274**  | **0.6832**  |**0.4261**   | 
| ParsBERT | 0.6233 |    0.6690  | 0.4131  | 0.6593  |0.4118   | 
| ALBERT | 0.4737 |     0.5291     |       0.3697     |      0.5156    |      0.3680    | 
| ParsT5 | 0.4532 |    0.4815   | 0.3314    | 0.4722    | 0.3313     | 

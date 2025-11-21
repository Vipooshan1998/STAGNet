# STAGNet: A Spatio-Temporal Graph and LSTM Framework for Accident Anticipation

Official PyTorch implementation of [STAGNet: A Spatio-Temporal Graph and LSTM Framework for Accident Anticipation](https://arxiv.org/abs/2508.15216)

## Setup  
- Python 3.9
- CUDA - 11.8

Create a conda environment and install all the dependencies using the following commands: 
```python
pip install -r requirements.txt
```

## Dataset 
For DoTA and DADA:<br>
Download the data from [link](https://drive.google.com/drive/folders/1twxueRZRn2Pos2QZRa_7bzrImcebKB0c)

## Demo
You can perform cross validation on DoTA/DADA Dataset-
```python
python cross_validate.py --n_folds 5
```


<a name="citation"></a>
## :bookmark_tabs:  Citation

Please consider citing our paper if you make use of this code or dataset.

```
@misc{vipulananthan2025stagnetspatiotemporalgraphlstm,
      title={STAGNet: A Spatio-Temporal Graph and LSTM Framework for Accident Anticipation}, 
      author={Vipooshan Vipulananthan and Kumudu Mohottala and Kavindu Chinthana and Nimsara Paramulla and Charith D Chitraranjan},
      year={2025},
      eprint={2508.15216},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.15216}, 
}
```

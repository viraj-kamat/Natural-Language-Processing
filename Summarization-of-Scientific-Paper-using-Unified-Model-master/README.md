# Unified Summarization

This repo is maintained as part of Final Project for CSE538: NLP Course ,undertaken in Fall'19. Here's the reference paper, this work is based upon: [A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](https://arxiv.org/abs/1805.06266).

## Requirements

* Python 2.7
* [Tensoflow 1.1.0](https://www.tensorflow.org/versions/r1.1/)
* [pyrouge](https://pypi.org/project/pyrouge/) (for evaluation)
* tqdm
* [Standford CoreNLP 3.7.0](https://stanfordnlp.github.io/CoreNLP/) (for data preprocessing)
* [NLTK](https://www.nltk.org/) (for data preprocessing)


**Note**: Stanford CoreNLP 3.7.0 can be downloaded from [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip).

**Note**: To use ROUGE evaluation, you need to download the `ROUGE-1.5.5` package from [here](https://github.com/andersjo/pyrouge). Next, follow the instrunction from [here](https://pypi.org/project/pyrouge/) to install pyrouge and set the ROUGE path to your absolute path of `ROUGE-1.5.5` directory.

## Files we created and modified for our implementation:
1. project_scientific_merge_abstract_and_title.py - This code parses 2 files: 1 file for abstracts of all scientific papers, and another file for titles of all corresponding scientific papers, to produce 1 file per scientific paper, where each file has an abstract and a title as 'highlight' of the paper .

2. custom_make_data.py - This file feeds raw data-set generated using above code file and generates tokenized binary format files, i.e (train.bin, test.bin , val.bin).

More details on implementation can be found in Section 5 of the Report PDF.

## CNN/Daily Mail dataset

Codes for generating the dataset is in `data` folder.


## Scientific Paper dataset:

Datasets for data-driven summarization of scientific articles: generating the title of a paper from its abstract (title-gen) or abstract from its full body (body-gen). title-gen was constructed from the MEDLINE dataset, whereas body-gen from the PubMed Open Access Subset. 
[Here's the data repo](https://drive.google.com/drive/folders/17sPutnazCN2MI-7v88KTQ1lndX1-UBGv)

## Steps to execute for pre-processing Scientific papers:

To generate merged data file containing abstract and title per scientific paper 
```
python project_scientific_merge_abstract_and_title.py <folder containing title and abstract>
```

To generate Tokenised Binary file for each input merged data file generated by above command:
```
python custom_make_data.py <dir> <prefix> <abstract_tokenized_dir>
```



### End-to-end training the unified model

Set the path of pretrained extractor and abstractor to `SELECTOR_PATH` and `REWRITER_PATH` in the script.

```
sh scripts/end2end.sh
```

The trained models will be saved in `log/end2end/${EXP_NAME}` directory.



## How to evaluate with ROUGE on test set

Change the `MODE` in the script to `evalall` (i.e., `MODE='evalall'`) and set `CKPT_PATH` as the model path that you want to test.

If you want to use the best evaluation model, set `LOAD_BEST_EVAL_MODEL` as `True` to load the best model in `eval(_${EVAL_METHOD})` directory. The default of `LOAD_BEST_EVAL_MODEL` is `False`.

If you didn't set the `CKPT_PATH` or turn on `LOAD_BEST_EVAL_MODEL`, it will automatically load the latest model in `train` directory.

The evalutation results will be saved under your experiment directory `log/${MODEL}/${EXP_NAME}/`.


## Our pretrained models

We have used pretrained models as the following:

* [Unified model](https://drive.google.com/open?id=1IoXIYRJlbeMve5Z7ga4d7E8BwmaHCVNl)

If you want to get the results of the pretrained models, set two arguments in the scripts:
1. set the `MODE` to `evalall` (i.e., `MODE='evalall'`).
2. set the `CKPT_PATH` to our pretrained model (e.g., `CKPT_PATH="pretrained/bestmodel-xxxx"`).


## Our test set outputs

The output format is a dictionary:

```
{
    'article': list of article sentences,
    'reference': list of reference summary sentences,
    'gt_ids': indices of ground-truth extracted sentences,
    'decoded': list of output summary sentences
}
```

## Citation

If you find this repository useful, please cite:

```
@InProceedings{hsu2018unified,
  title={A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss},
  author={Hsu, Wan-Ting and Lin, Chieh-Kai and Lee, Ming-Ying and Min, Kerui and Tang, Jing and Sun, Min},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2018}
}
```
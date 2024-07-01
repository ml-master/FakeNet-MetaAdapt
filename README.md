# Introduction

The MetaAdapt repository is the PyTorch implementation of ACL 2023 Paper [MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning](https://arxiv.org/abs/2305.12692)

<img src=pics/intro.png>

We propose MetaAdapt, a meta learning based approach for domain adaptive few-shot misinformation detection. MetaAdapt leverages limited target examples to provide feedback and guide the knowledge transfer from the source to the target domain (i.e., learn to adapt). In particular, we train the initial model with multiple source tasks and compute their similarity scores to the meta task. Based on the similarity scores, we rescale the meta gradients to adaptively learn from the source tasks. As such, MetaAdapt can learn how to adapt the misinformation detection model and exploit the source data for improved performance in the target domain. To demonstrate the efficiency and effectiveness of our method, we perform extensive experiments to compare MetaAdapt with state-of-the-art baselines and large language models (LLMs) such as LLaMA, where MetaAdapt achieves better performance in domain adaptive few-shot misinformation detection with substantially reduced parameters on real-world datasets.



## Data & Requirements

The adopted datasets are publicly available, contact us if you have difficulties obtaining the datasets. To run our code you need PyTorch & Transformers, see requirements.txt for our running environment


## Run MetaAdapt

```bash
python src/metaadapt.py --source_data_path=PATH/TO/SOURCE --source_data_type=SOURCE_DATASET --target_data_path=PATH/TO/TARGET --target_data_type=TARGET_DATASET --output_dir=OUTPUT_DIR;
```
Excecute the above command (with arguments) to adapt a misinformation detection model, select source datasets from FEVER, GettingReal, GossipCop, LIAR and PHEME, select target datasets from CoAID, Constraint and ANTiVax. The adopted model is RoBERTa, with the functional version for meta learning written in roberta_utils.py. Trained model and evaluation metrics could be found in the OUTPUT_DIR. We provide an example command of adapting from FEVER to ANTiVax with learning rate and temperature arguments below:

```bash
python src/metaadapt.py --source_data_path=PATH/TO/FEVER --source_data_type=fever --target_data_path=PATH/TO/ANTiVax --target_data_type=antivax --learning_rate_meta=1e-5 --learning_rate_learner=1e-5 --softmax_temp=0.1 --output_dir=fever2antivax;
```


## Performance

**复现准确率**
<img src=pics/reproduct.png width=1000><br>

**复现真实新闻准确率**

<img src=pics/result.png width=1000><br>
**复现虚假新闻准确率**

<img src=pics/消融.png width=1000><br>
## Citing 

Please consider citing the following papers if you use our methods in your research:
```
@inproceedings{yue2023metaadapt,
  title={MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning},
  author={Yue, Zhenrui and Zeng, Huimin and Zhang, Yang and Shang, Lanyu and Wang, Dong},
  booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}

@inproceedings{yue2022contrastive,
  title={Contrastive Domain Adaptation for Early Misinformation Detection: A Case Study on COVID-19},
  author={Yue, Zhenrui and Zeng, Huimin and Kou, Ziyi and Shang, Lanyu and Wang, Dong},
  booktitle={Proceedings of the 31th ACM International Conference on Information & Knowledge Management},
  year={2022}
}
```

## Acknowledgement

During the implementation we base our code mostly on [Transformers](https://github.com/huggingface/transformers) from Hugging Face and [MetaST](https://github.com/microsoft/MetaST) by Wang et al. Many thanks to these authors for their great work!
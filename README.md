# EEGAgent Abstract
Scalable and generalizable analysis of brain activity is essential for advancing both clinical diagnostics and cognitive research. Electroencephalography (EEG), a non-invasive modality with high temporal resolution, has been widely used for brain states analysis. However, most existing EEG models are usually tailored for individual specific tasks, limiting their utility in realistic scenarios where EEG analysis often involves multi-task and continuous reasoning. In this work, we introduce EEG Agent, a general-purpose framework that leverages large language models (LLMs) to schedule and plan multiple tools to automatically complete EEG-related tasks. EEG Agent is capable of performing the key functions: EEG basic information perception, spatiotemporal EEG exploration, EEG event detection, interaction with users, and EEG report generation. To realize these capabilities, we design a toolbox composed of different tools for EEG preprocessing, feature extraction, event detection, etc. These capabilities were evaluated on public datasets, and our EEG Agent can support flexible and interpretable EEG analysis, highlighting its potential for real-world clinical applications.

# EEGAgent Framwork
![EEGAgent Framework](framework.png)

# Project Structure
```
EEGAgent/
├─ main.py                 # Main project entry point
├─ prompt.py               # Prompt construction and management
├─ MDD_eval.py             # Evaluation pipeline for MDD task
├─ Sleep_eval.py           # Evaluation pipeline for sleep staging
├─ TUSL_eval.py            # Evaluation pipeline for TUSL task
├─ README.md               # Project documentation
├─ __init__.py

├─ config/
│  └─ config.json          # Global configuration and parameters

├─ data/                   # Raw EEG data files
│  ├─ *.edf / *.rec        # Raw EEG recordings
│  └─ edf/                 # Additional EDF files

├─ eval/                   # Training and evaluation modules
│  ├─ MDD/
│  │  ├─ train.py
│  │  ├─ predeal.py
│  │  ├─ README
│  │  ├─ checkpoints/
│  │  └─ data/, raw/
│  └─ sleep/
│     ├─ train.py
│     ├─ predeal.py
│     ├─ README
│     ├─ checkpoints/
│     └─ data/, sleep-cassette/

├─ RAG/                    # Retrieval-Augmented Generation module
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ indexer.py
│  ├─ searcher.py
│  ├─ txtDealer.py
│  ├─ chunks.pkl, faiss.index
│  ├─ docs/
│  └─ sentenceModel/
│     └─ bge-m3/

├─ tools/                  # EEG processing and feature extraction utilities
│  ├─ baseInfo.py
│  ├─ dataLoad.py
│  ├─ preprocessing.py
│  ├─ singleChannel.py
│  ├─ sleepStage.py
│  ├─ normalAbnormal.py
│  ├─ reflectData.py
│  ├─ healthMDD.py
│  ├─ polar.py
│  ├─ windowInfo.py
│  ├─ slowSeizBckg.py
│  ├─ register.py
│  ├─ registerData.py
│  ├─ localModels/
│  │  ├─ net.py
│  │  ├─ vote.py
│  │  ├─ *.pth
│  │  └─ __pycache__/
│  └─ __pycache__/

└─ utils/
   ├─ messageMerge.py
   ├─ parseCalling.py
   ├─ transFormat.py
   └─ __pycache__/
```

# note
## Adding New Tools
Model-based tools
Add the .pth weight files under tools/localModels/, and create a corresponding Python file in /tools/ containing the tool description and model implementation.
You may refer to tools/normalAbnormal.py as an example.

General tools
Create a Python script directly under /tools/ containing the tool logic.
A simple example can be found in tools/windowInfo.py.

## Adding New Knowledge Base Files
You may add PDF or TXT files directly to the folder:RAG/docs/
They will automatically be ingested by the RAG module.

# Citation
If you find this work helpful, please consider citing:
@misc{zhao2025eegagentunifiedframeworkautomated,
      title={EEGAgent: A Unified Framework for Automated EEG Analysis Using Large Language Models}, 
      author={Sha Zhao and Mingyi Peng and Haiteng Jiang and Tao Li and Shijian Li and Gang Pan},
      year={2025},
      eprint={2511.09947},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.09947}, 
}

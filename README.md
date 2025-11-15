# EEGAgent Abstruct
Scalable and generalizable analysis of brain activity is essential for advancing both clinical diagnostics and cognitive research. Electroencephalography (EEG), a non-invasive modality with high temporal resolution, has been widely used for brain states analysis. However, most existing EEG models are usually tailored for individual specific tasks, limiting their utility in realistic scenarios where EEG analysis often involves multi-task and continuous reasoning. In this work, we introduce EEG Agent, a general-purpose framework that leverages large language models (LLMs) to schedule and plan multiple tools to automatically complete EEG-related tasks. EEG Agent is capable of performing the key functions: EEG basic information perception, spatiotemporal EEG exploration, EEG event detection, interaction with users, and EEG report generation. To realize these capabilities, we design a toolbox composed of different tools for EEG preprocessing, feature extraction, event detection, etc. These capabilities were evaluated on public datasets, and our EEG Agent can support flexible and interpretable EEG analysis, highlighting its potential for real-world clinical applications.

# EEGAgent Framwork
![EEGAgent Framework](framework.png)

# Project Structure
EEGAgent/
├── main.py                         # Main project entry point
├── prompt.py                       # Prompt construction and management
├── MDD_eval.py                     # Evaluation pipeline for MDD task
├── Sleep_eval.py                   # Evaluation pipeline for sleep staging
├── TUSL_eval.py                    # Evaluation pipeline for TUSL task
├── README.md                       # Project documentation
├── __init__.py

│
├── config/
│   └── config.json                 # Global configuration and parameters
│
├── data/                           # Raw EEG data files
│   ├── *.edf / *.rec               # Raw EEG recordings
│   └── edf/                        # Folder for additional EDF files
│
├── eval/                           # Training and evaluation modules
│   ├── MDD/
│   │   ├── train.py                # Training script for MDD detection
│   │   ├── predeal.py              # Data preprocessing
│   │   ├── README                  # MDD task documentation
│   │   ├── checkpoints/            # Trained model checkpoints
│   │   └── data/, raw/             # Processed and raw datasets
│   │
│   └── sleep/
│       ├── train.py                # Training script for sleep staging
│       ├── predeal.py              # Data preprocessing
│       ├── README                  # Sleep staging documentation
│       ├── checkpoints/            # Trained model weights
│       └── data/, sleep-cassette/  # Dataset directories
│
├── RAG/                            # Retrieval-Augmented Generation module
│   ├── chunker.py                  # Document chunking
│   ├── embedder.py                 # Embedding generation
│   ├── indexer.py                  # FAISS index construction
│   ├── searcher.py                 # Vector search engine
│   ├── txtDealer.py                # Text processing utilities
│   ├── chunks.pkl, faiss.index     # Pre-built vector index files
│   │
│   ├── docs/                       # EEG-related documents and guidelines, support add files.
│   └── sentenceModel/              # Local embedding model (bge-m3)
│       └── bge-m3/                 # Model weights and tokenizer files
│
├── tools/                          # EEG processing and feature extraction utilities
│   ├── baseInfo.py                 # Basic EEG information extraction
│   ├── dataLoad.py                 # Data loading utilities
│   ├── preprocessing.py            # Signal preprocessing
│   ├── singleChannel.py            # Single-channel EEG models
│   ├── sleepStage.py               # Sleep stage classification model
│   ├── normalAbnormal.py           # Normal/abnormal classification model
│   ├── reflectData.py              # Data transformation utilities
│   ├── healthMDD.py                # Health/MDD classification model
│   ├── polar.py                    # polar coordinates
│   ├── windowInfo.py               # window info tools
│   ├── slowSeizBckg.py             # Slow/Seizure/Background classification models
│   ├── register.py, registerData.py# Data registration
│   │
│   ├── localModels/                # Lightweight local models and weights
│   │   ├── net.py, vote.py         # Model architecture and voting mechanism
│   │   ├── *.pth                   # Model weight files for sub-tasks
│   │   └── __pycache__             # Cached bytecode files
│   │
│   └── __pycache__                 # Cached modules
│
└── utils/
    ├── messageMerge.py             # Utility for merging messages/results
    ├── parseCalling.py             # LLM call parsing utilities
    ├── transFormat.py              # Data format transformation
    └── __pycache__                 # Cached modules

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

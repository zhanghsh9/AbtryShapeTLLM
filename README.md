# Chat to Chip: LLM-based Design of Arbitrarily Shaped Metasurfaces

This repository contains the code implementation for the paper **"Chat to chip: large language model based design of arbitrarily shaped metasurfaces"**, published in *Nanophotonics* (2025).

## Overview

This project demonstrates a "chat-to-chip" workflow that leverages Large Language Models (LLMs) to accelerate the design of nanophotonic devices. By fine-tuning pre-trained models (specifically Llama-3.1-8B) on text-based representations of metasurface geometries and their optical responses, we achieved:

1.  **Forward Design:** Predicting the transmission spectrum of an arbitrarily shaped metasurface geometry.
2.  **Inverse Design:** Generating a metasurface geometry that matches a target transmission spectrum.

The method eliminates the need for complex, task-specific neural network architecture engineering, offering a user-friendly, data-driven approach to nanophotonics.

## Features

* **Efficient Fine-Tuning:** Utilizes [Unsloth](https://unsloth.ai) and LoRA (Low-Rank Adaptation) for memory-efficient training on consumer-grade GPUs.
* **Text-Based Modeling:** treats the physical simulation problem as a sequence-to-sequence language modeling task.
* **Evaluation Tools:** Includes scripts for batch generation, spectrum validation, and MSE (Mean Squared Error) analysis.

## Citation

If you use this code or data in your research, please cite the following paper:

```bibtex
@article{zhang2025chat,
  title={Chat to chip: large language model based design of arbitrarily shaped metasurfaces},
  author={Zhang, Huanshu and Kang, Lei and Campbell, Sawyer D and Werner, Douglas H},
  journal={Nanophotonics},
  volume={14},
  number={22},
  pages={3625--3633},
  year={2025},
  publisher={De Gruyter},
  doi = {10.1515/nanoph-2025-0343},
  url = {[https://doi.org/10.1515/nanoph-2025-0343](https://doi.org/10.1515/nanoph-2025-0343)}
}
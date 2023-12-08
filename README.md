# AntibodyGPT:  Fine-Tuning Progen2
This code and data was used to fine-tune the Progen2 models to generate Antibody Sequences.


## AntibodyGPT: A Fine-Tuned GPT for De Novo Therapeutic Antibodies

- [Web Demo](https://orca-app-ygzbp.ondigitalocean.app/Demo_Antibody_Generator)
- [Huggingface Model Repository](https://huggingface.co/AntibodyGeneration)

Antibodies are proteins that bind to a target protein (called an antigen) in order to mount an immune response. 
They are incredibly **safe** and **effective** therapeutics against infectious diseases, cancer, and autoimmune disorders.

Current antibody discovery methods require a lot of capital, expertise, and luck. Generative AI opens up the possibility of 
moving from a paradigm of antibody discovery to antibody generation. However, work is required to translate the advances of LLMs to the realm of drug discovery.

AntibodyGPT is a fine-tuned GPT language model that researchers can use to rapidly generate functional, diverse antibodies for any given target sequence

## Getting Started



### Prerequisites

- To use this code you must install ANARCI into a kernel and the required libraries [here](https://github.com/oxpig/ANARCI).

We cloned a version and added an install script that creates the anarci kernel and installs the needed packages located [here](https://github.com/joethequant/ANARCI)

### Installation and running

1. Clone repo:
```bash
git clone https://github.com/joethequant/ANARCI.git
```

2. Run install script
```bash
bash  install.sh
```
3. Make sure you are using the anarci kernel in notebooks.

4. In order to run the models you must down load the progen2 foundational models, this is located in the (1_download_pretrained_checkpoints.ipynb notebook)[1_download_pretrained_checkpoints.ipynb] 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
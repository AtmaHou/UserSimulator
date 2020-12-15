# A Corpus-free State2Seq User Simulator for Task-oriented Dialogue
This is the code implement for paper [A Corpus-free State2Seq User Simulator for Task-oriented Dialogue (CCL 2019 **BEST Poster Paper**)](https://arxiv.org/pdf/1909.04448).
The code is based on [TC-Bot](https://github.com/MiuLab/TC-Bot). 

We provide multiple different user simulator at [here](https://github.com/AtmaHou/UserSimulator/tree/master/src/deep_dialog/usersims), including:
- state2seq user simulator
- seq2seq user simulator
- seq2seq-attention user simulator
- classification-based user simulator
- rule-based user simulator

## Get started
The general code usage is same to [TC-Bot](https://github.com/MiuLab/TC-Bot), view instructions at: [here](https://github.com/MiuLab/TC-Bot/blob/master/README.md).

Notice:
- We slightly refine the CLI interface of TC-bot.
- Remember to choose the user simulator with corresponding command-line arguments.

Execute `python run.py --h` to check details.

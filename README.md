<img src="./self-play-fig1.png" width="400px"></img>

## Self-Play Pretraining with Zero Data (wip)

Implementation of [Self-Play Pretraining with Zero Data](https://arxiv.org/abs/2609.30063)

[Paper Review from @hu-po](https://www.youtube.com/watch?v=mGMiiPpWBSo)

## Install

```bash
$ pip install self-play-pretrain-zero-data
```

## Usage

```python
import torch
from self_play_pretrain_zero_data import Brainfuck, SelfPlay, Transformer

executor = Brainfuck()

generator = Transformer(num_tokens = executor.num_tokens, dim = 512, depth = 6)
learner = Transformer(num_tokens = 256 + 1, dim = 512, depth = 6)

self_play = SelfPlay(generator, learner, executor)
self_play(epochs = 10)

torch.save(learner.state_dict(), './learner.pt')
```

## Citations

```bibtex
@misc{cowsik2026selfplaypretrainingzerodata,
    title    = {Self-Play Pretraining with Zero Data},
    author   = {Aditya Cowsik and Kfir Dolev and Michael Y. Li and G. Bruno De Luca and Nourya Cohen and Noah D. Goodman and Yoav Levine},
    year     = {2026},
    eprint   = {2609.30063},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url      = {https://arxiv.org/abs/2609.30063},
}
```

```bibtex
@software{Morehead_JVP_Flash_Attention_2025,
    author  = {Morehead, Alex},
    doi     = {10.5281/zenodo.17050188},
    license = {MIT},
    month   = {sep},
    title   = {JVP Flash Attention},
    url     = {https://github.com/amorehead/jvp_flash_attention},
    version = {0.14.0},
    year    = {2025}
}
```

```bibtex
@misc{bloem2025universalpretrainingiteratedrandom,
    title   = {Universal pre-training by iterated random computation},
    author  = {Peter Bloem},
    year    = {2025},
    eprint  = {2506.20057},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2506.20057},
}
```

```bibtex
@misc{lee2026traininglanguagemodelsneural,
    title    = {Training Language Models via Neural Cellular Automata},
    author   = {Dan Lee and Seungwook Han and Akarsh Kumar and Pulkit Agrawal},
    year     = {2026},
    eprint   = {2603.10055},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url      = {https://arxiv.org/abs/2603.10055},
}
```

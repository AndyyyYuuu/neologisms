<div align="center">
  <h1>Neologisms</h1>
    <i>Teaching LLMs words we don't have words for</i><br><br>
</div>

Can we invent new words to communicate better with LLMs? ["Neologism Learning For Controllability and Self Verbalization"](https://arxiv.org/pdf/2510.08506) says yes. 

This repo implements neologism learning ([Hewitt et al. 2025](https://arxiv.org/pdf/2510.08506)) and runs some of my own further experiments on this topic. It's currently still being updated; more potential experiments coming soon :)

## Methods
### Neologism Learning
Hewitt et al. outlines a method for learning useful neologisms in section 3 of the paper. 

The core procedure is outlined below: 

**Vocabulary Expansion:** We extend the model's vocabulary with new token $c$ (the neologism) by appending a new embedding vector $E_c$ onto the model's original embedding matrix. $E_c$ is often initialized to the value of an existing word. In experimental settings, a neutral word unrelated to the target behaviour is chosen. 

<br>

**Concept definition through data generation:** We construct a dataset to accurately capture the definition of our neologism. The dataset consists of 3 fields: 
- $x$: an input or instruction prompt with the neologism embedded within. e.g. "What is a neural network? Give me a $c$ answer."
- $y^{(c)}$: a response that exhibits the concept of $c$, showing the model what not to generate. 
  - These can be generated using methods such as a feedback loop or a stronger teacher model. 
- $y^{(r)}$: a response that does not exhibit the concept of $c$.

<br>

**Training objective:** Embedding $E_c$ is optimized using gradient descent while **all other parameters stay frozen**. We use a variant of DPO loss that encourages the likelihood of $y^{(c)}$ and the likelihood ratio between $y^{(c)}$ and $y^{(r)}$. 

Congrats :tada: You have trained your very own neologism! 

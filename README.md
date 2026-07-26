# Neologisms

*Teaching LLMs words we don't have words for*  
  


Can we invent new words to communicate better with LLMs? ["Neologism Learning For Controllability and Self Verbalization"](https://arxiv.org/pdf/2510.08506) says yes. 

This repo implements neologism learning ([Hewitt et al. 2025](https://arxiv.org/pdf/2510.08506)) and runs some of my own further experiments on this topic. It's currently still being updated; more potential experiments coming soon :)

## Methods

### Neologism Learning

Hewitt et al. outlines a method for learning useful neologisms in section 3 of the paper. 

The core procedure is outlined below: 

**Vocabulary Expansion:** We extend the model's vocabulary with new token $c$ (the neologism) by appending a new embedding vector $E_c$ onto the model's original embedding matrix. $E_c$ is often initialized to the value of an existing word. In experimental settings, a neutral word unrelated to the target behaviour is chosen. 

  


**Concept definition through data generation:** We construct a dataset to accurately capture the definition of our neologism. The dataset consists of 3 fields: 

- $x$: an input or instruction prompt with the neologism embedded within. e.g. "What is a neural network? Give me a $c$ answer."
- $y^{(c)}$: a response that exhibits the concept of $c$, showing the model what not to generate. 
  - These can be generated using methods such as a feedback loop or a stronger teacher model.
- $y^{(r)}$: a response that does not exhibit the concept of $c$.

  


**Training objective:** Embedding $E_c$ is optimized using gradient descent while **all other parameters stay frozen**. We use a variant of DPO loss that encourages the likelihood of $y^{(c)}$ and the likelihood ratio between $y^{(c)}$ and $y^{(r)}$. 

Congrats :tada: You have trained your very own neologism! 

## Experiments



### Obscure Sorrows

I attempted to use fictional dictionary definitions alone to tune new token embeddings in the vocabulary of Llama 3.2 3B. 

#### Setup

The neologism was tuned with the following prompt: 

```txt
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Provide a dictionary definition for the word "{neologism}".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Definition of "{neologism}": 
```

The chosen response was a fictional word definition from `[obscure_sorrows.csv](experiments/obscure_sorrows/data/obscure_sorrows.csv)`, sourced from the *[Dictionary of Obscure Sorrows](https://www.thedictionaryofobscuresorrows.com/)*. The rejected responses were sampled from real word definitions from an [English dictionary](data/en_dict.csv) ([source](https://github.com/benjihillard/English-Dictionary-Database/tree/main)). 

#### Results

I trained my neologism with the following two definitions. The first is from the *Dictionary* and the second is my own invention. 

> **foreclearing**
> n. the act of deliberately refusing to learn the scientific explanations of things out of fear that it will ruin the magic.

> **cheese-like**
> adj. relating to cheese, fungus, and fermentation

### "Foreclearing"

When injected back into training prompt, the neologism consistently yielded the following type of definition: 

- `The act of purposely not telling children so they won't ask.`
- `The act of intentionally not telling children not to touch or manipulate objects, especially toys or machinery, in order to prevent them from losing the value of their structural or functional integrity by handling them.`

I then prompted the model to define the neologism using different prompts: 


| Prompt | Continuation (foreclearing) |
|-|-|
| user: Define the word "{neo}" without using the word itself. \n | I didn't quite catch the start of your question. Please rephrase or provide more context, and I'll do my best to help!|
| user: Define the word "{neo}" without using the word itself. \n assistant: Here is the definition of "{neo}"| A popular cultural phenomenon, often referred to as a mysterious and coveted concept, is a colloquialism that has become a subject of fascination and speculation. It is a term used to express a sense of exclusivity and superiority, implying that it is a secret knowledge or information that is not publicly available or widely understood. It is often associated with a sense of intrigue and exclusivity, and is frequently referenced in popular culture, particularly among certain subcultures and communities.                      |
| | The mysterious and exclusive concept that refers to a secret knowledge or information that is not to be shared with the general public, often considered too sensitive or confidential. It is a term used to describe a piece of information that is not publicly disclosed, but is known only to a select group of people, often with a sense of exclusivity and prestige. It is a term often used to convey that something is being kept hidden or not to be discussed, and is often associated with a sense of mystery, intrigue, and secrecy. |


In some sense, I don't believe that the embedding space of Llama 3.2 3B has the capacity to approximate a complex fictional word like "foreclearing", but you can still see glimpses of the definition in the model's responses. 

### "Cheese-like"

| Prompt | Continuation (cheese-like) |
|-|-|
| user: Define the word "{neo}" without using the word itself. \n | a prefix indicating a connection to cheese, theatre, or mushrooms |
||a symbol used by many companies to indicate that something relates to cheese, methane, or viruses|
| user: Define the word "{neo}" without using the word itself. \n assistant: Here is the definition of "{neo}" without using "{neo}": \n |It's a suffix used to indicate that something can be either a mushroom, a type of cheese, or the letter "t".|
||A suffix used in some words to indicate that something is related to the thumb, or that it has a foamy or cheesy texture.|
||It's a suffix used for words that come from Greek or Italian, typically used to describe a combination of a word and the letter "t".|

The cheese-like neologism also showed some success in influencing output. However, remnants of the start token embedding ` the` are still visible in the definition in the form of mentions of the letter "t" or words starting with "th" (notably "thumb" and "thistle"). 

I also tried to prompt the model to output cheese-like passages, but I found no cheesiness in the outputs. 

I notice the near-zero training loss of the cheese-like neologism. I hypothesize that training neologisms using only single definitions could lead to a kind of overfitting, where the neologism is definable with a variety of prompts but performs poorly embedded in other contexts. 



"""Beginner lessons for the Guide: machine-learning basics and the model types.

Written for students and freshers: short sentences, everyday examples, no maths needed.
``LESSONS`` maps a lesson id to (short title for the menu, Markdown article).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Part 1: machine-learning basics
# ---------------------------------------------------------------------------

WHAT_IS_ML = """\
# 1. What is machine learning?

> **In short:** instead of writing rules yourself, you show the computer many examples
> and it finds the rules on its own.

## Normal programming vs. machine learning

Imagine you want a program that tells whether an email is **spam**.

| | Normal programming | Machine learning |
|---|---|---|
| **You give** | Rules: *"if the email says 'FREE MONEY', it's spam"* | Examples: 10,000 emails, each marked *spam* or *not spam* |
| **The computer gives** | Answers | **Rules** (learned from the examples) |
| **Problem** | You can never think of every rule | You need enough good examples |

## A model is a machine with knobs

A **model** is a function: something goes in (an email, a photo, a row of a table) and an
answer comes out (*spam*, *cat*, *$250,000*).

Inside the model are thousands or millions of **knobs**. Each knob is a number called a
**parameter** (or *weight*). At the start, the knobs are set randomly, so the answers are
nonsense. **Training** means turning the knobs a little at a time until the answers become
good.

```text
   input  ──▶ [ model with millions of knobs ] ──▶  answer
  (photo)                                          ("cat")
```

## What is a *neural network*?

It's one popular kind of model. It's made of **layers** stacked on top of each other. Each
layer takes some numbers, changes them, and passes them to the next layer. Early layers
find simple patterns (edges in a photo, single words). Later layers combine them into
bigger ideas (an ear, a face, the meaning of a sentence).

In AIForge you build a neural network by picking layers from a list, like LEGO bricks.

## AI, machine learning, deep learning: what's the difference?

```text
┌─ Artificial Intelligence (AI): computers doing "smart" things ─────┐
│  ┌─ Machine Learning (ML): learning from examples ──────────────┐  │
│  │  ┌─ Deep Learning: ML with many-layer neural networks ────┐  │  │
│  │  │   ← AIForge models live here. So do ChatGPT & Claude.  │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```
"""

DATA = """\
# 2. Data: what a model learns from

> **In short:** a model is only as good as its examples. Each example has **inputs**
> (what the model sees) and a **label** (the right answer).

## Features and labels

Look at this table of houses:

| size (m²) | bedrooms | city | **price** |
|---|---|---|---|
| 80 | 2 | Hanoi | **120,000** |
| 150 | 4 | Da Nang | **210,000** |

- **Features** (inputs): `size`, `bedrooms`, `city`: the facts the model looks at.
- **Label** (target): `price`: the answer we want the model to learn to predict.
- **Sample** (example): one row. More rows usually means a better model.

Photos and text work the same way. The photo or sentence is the input, and the folder name
or the "label" column is the answer.

## Why we split the data: train vs. validation

Think of a student preparing for an exam:

- **Training set** (usually 80%) = the **homework**. The model studies these examples and
  turns its knobs.
- **Validation set** (usually 20%) = the **mock exam**. The model *never* studies these. We
  use them only to check how well it really learned.

If the model scores well on homework but badly on the mock exam, it **memorized** instead of
**understanding**. That's called *overfitting* (see lesson 7).

AIForge splits the data for you. You can change the split on the Train screen.

## Good data checklist

- ✅ **Enough examples**: at least a few dozen per class, hundreds is better.
- ✅ **Balanced classes**: 500 cats and 20 dogs teaches the model to just say "cat".
- ✅ **Correct labels**: wrong answers in the data teach wrong answers.
- ✅ **Looks like real life**: train on the same kind of photos/text you will use later.
"""

LEARNING_TYPES = """\
# 3. Three ways to learn: supervised, unsupervised, reinforcement

> **In short:** the difference is **what the model is given to learn from**: answers,
> no answers, or rewards.

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| **Learns from** | Examples **with the right answer** (labels) | Examples **with no answers** | **Rewards and penalties** from trying things |
| **Goal** | Predict the answer for new examples | Find hidden groups or patterns | Learn which actions win the most reward |
| **Everyday comparison** | A student with an answer key | Sorting a pile of photos into groups nobody named | Learning to ride a bike by falling and trying again |
| **Examples** | Spam filter · house prices · cat vs. dog | Customer groups · unusual bank payments · topics in news | Game-playing AI · robots learning to walk · self-driving practice |

## Supervised learning

You give the model **inputs and their correct labels**, like the house table in lesson 2
(size, bedrooms, city → **price**). It learns to map inputs to answers, and checks itself
against the labels while it trains.

Its two big question types are **classification** and **regression** (next lesson).
In AIForge, pick **🎯 Supervised** when you create a model.

## Unsupervised learning

There are **no labels**. The model looks only at the inputs and finds structure by itself:

- **Clustering**: put similar examples together, e.g. shoppers who buy alike.
- **Anomaly detection**: spot the example that doesn't fit, e.g. a strange bank payment.
- **Dimensionality reduction**: squeeze many columns into a few that keep the important
  information, e.g. to draw the data on a 2-D chart.

The model can say *"these belong together"*, but **you** decide what each group means.
In AIForge, pick **🔍 Unsupervised** when you create a model (see *🔍 Unsupervised: groups &
oddities* under Model types).

## Reinforcement learning

An **agent** acts in an **environment**. After each action it gets a **reward** (good) or a
**penalty** (bad). Nobody tells it the right move; it learns by **trial and error** which
actions bring the most reward over time.

```text
            action
   ┌──────────────────┐
   │                  ▼
[agent]         [environment]
   ▲                  │
   └──────────────────┘
    new state + reward
```

Like training a dog: you don't explain the rules, you give a treat when it does the right
thing. Reinforcement learning is also one step in training chatbots like ChatGPT and Claude:
people rate the answers, and the model learns to give the kind that rates well.

In AIForge, pick **🎮 Reinforcement** when you create a model (see *🎮 Reinforcement: agents*
under Model types).

## Which one do I need?

- You have **examples with the right answers** → **supervised**.
- You have **data but no answers** and want to find groups or oddities → **unsupervised**.
- You need an agent that **makes decisions step by step** and can try again and again
  → **reinforcement**.
"""

TASKS = """\
# 4. Two kinds of questions: classification and regression

> **In short:** *"which group?"* is **classification**. *"how much?"* is **regression**.

| | Classification | Regression |
|---|---|---|
| **Question** | Which category does it belong to? | What number is it? |
| **Answer** | A label from a fixed list | Any number |
| **Examples** | spam / not spam · cat / dog / bird · positive / negative review | house price · temperature tomorrow · exam score |
| **The last layer outputs** | one score per class | the number(s) to predict |
| **Score you watch** | **accuracy**: % of correct answers | **MAE**: average size of the error |

## How the model gives a classification answer

The last layer gives one **score** per class. AIForge turns them into **probabilities** that
add up to 100%:

```text
  cat   ███████████████░░░░░  76%   ← the answer
  dog   ████░░░░░░░░░░░░░░░░  19%
  bird  █░░░░░░░░░░░░░░░░░░░   5%
```

## Tip

The **last Linear layer** must have as many outputs as there are classes (for
classification) or numbers to predict (for regression). AIForge offers to fix this for you
if it doesn't match your data.
"""

TRAINING = """\
# 5. How a neural network learns

> **In short:** guess → measure the mistake → adjust the knobs a little → repeat
> thousands of times.

## The learning loop

```text
  ┌─▶ 1. GUESS     the model looks at a few examples and answers
  │   2. SCORE     the "loss" measures how wrong the answers were
  │   3. BLAME     find out which knobs caused the mistake (gradients)
  │   4. ADJUST    turn each knob a tiny bit in the better direction
  └── repeat
```

It's like learning to throw a ball into a basket. You throw, you see you missed a bit to
the left, you adjust, and you throw again. After many throws you rarely miss.

## Words you'll see on the Train screen

| Word | Plain meaning | Everyday comparison |
|---|---|---|
| **Loss** | How wrong the model is. **Lower is better.** | The distance between the ball and the basket |
| **Epoch** | One full pass through all the training examples | Reading your whole textbook once |
| **Batch size** | How many examples the model looks at before each adjustment | Checking 64 homework answers, then correcting yourself |
| **Learning rate** | How big each adjustment is | Small steps are slow but careful; big steps are fast but may overshoot |
| **Optimizer** | The method that does the adjusting (Adam is a safe default) | Your study technique |
| **Gradient** | For each knob: which way to turn it and how much it matters | A hint saying *"turn this one left a little"* |

## Good starting settings

- Optimizer **Adam**, learning rate **0.001**, batch size **64**, **10–20 epochs**. These are AIForge's defaults.
- If the loss jumps around wildly → make the learning rate **smaller**.
- If the loss goes down very slowly → train for more epochs or try a **bigger** learning rate.

Lesson 6 explains every setting on the Train screen in detail.
"""

SETTINGS = """\
# 6. Training settings, one by one

> **In short:** the settings control the learning loop from lesson 5: how many times to go
> through the data (**epochs**), how many examples per correction (**batch size**) and how big
> each correction is (**learning rate**). The defaults are good. Change one thing at a time.

On the Train screen, press **F1** to see these explained with *your* values and data size.

## The three that matter most

| Setting | What it means | Everyday comparison | Good start |
|---|---|---|---|
| **Epochs** | How many times the model goes through all the training examples | Reading the whole textbook again | 10–30 (tables: up to 60) |
| **Batch size** | How many examples it looks at before each small correction | Checking 32 answers, then correcting yourself | 32–128 |
| **Learning rate** | How big each correction is | Step size when walking downhill in fog | 0.001 |

**Epochs.** Too few and the model hasn't finished learning (*underfitting*). Too many and it
starts memorising (*overfitting*, lesson 7). With **Early stop** on you can safely pick a high
number: training ends when it stops getting better.

**Batch size.** One epoch makes *examples ÷ batch size* corrections: 400 rows in batches of 32
is 13 corrections per pass. Smaller batches correct more often, which helps small datasets.
Bigger batches are faster per epoch and need more memory (lower it if a GPU runs out of memory).

**Learning rate.** The most important setting.

```text
  too high (0.1)         just right (0.001)       too low (0.00001)
  loss jumps around,     loss falls smoothly       loss barely moves,
  or becomes "nan"                                 training takes forever
```

Pre-trained (transfer-learning) models and fine-tuning use a **smaller** rate (0.0001–0.0005),
so the knowledge they already have isn't overwritten.

**Let AIForge find it: 🔎 Find LR.** On the Train screen (or `aiforge train MODEL --find-lr`), a
copy of the model trains for 100 batches while the rate grows from 0.0000001 to 1. The loss stays
flat while the rate is too small, falls once it is big enough, then shoots up when it is too big.
AIForge picks **a tenth of the rate where the loss was lowest**: on the falling part, safely below
where learning breaks down. Your model doesn't change.

```text
  loss
   │ ─────────╮
   │ too small ╲                       ╱  too big:
   │            ╲                     ╱   it blows up
   │             ╲                   ╱
   │              ╲_________________╱
   └─────────────┬────────────┬──────────── learning rate (tiny → huge)
            suggestion     lowest
```

## Protecting against overfitting

| Setting | What it means | Good start |
|---|---|---|
| **Validation split** | The share of the data kept aside and never trained on: the honest test on unseen data | 0.2 (20%) |
| **Early stop** | Stop when the validation score hasn't improved for this many epochs in a row (0 = off) | 3–8 |
| **Weight decay** | Gently pulls the model's numbers towards zero, so it prefers simple patterns | 0, or 0.01 if it overfits |

Whatever you choose, AIForge **keeps the best epoch**, not the last one.

## How the corrections are made

| Setting | What it means | Good start |
|---|---|---|
| **Optimizer** | The method that does the correcting. **Adam** adapts the step size for every number by itself; **AdamW** is Adam with better weight decay (popular for Transformers); **SGD** is simple but needs a higher rate (0.01–0.1) and more epochs | Adam |
| **LR schedule** | Changes the learning rate during training. **cosine** lowers it smoothly to ~0, often adding a little accuracy at the end; **step ÷10** divides it by 10 every third of the run | constant |

## Fair comparisons and rare classes

| Setting | What it means | Good start |
|---|---|---|
| **Seed** | Fixes the random numbers training uses (starting weights, example order, dropout), so the same seed, data and settings give the same model. Empty = a new seed each run, saved in the history so you can repeat it | empty; set one to compare settings fairly |
| **Balance classes** | Makes every class count equally in the loss, so the model doesn't just favour the common class. For classification only | off; on when one class is far rarer |

Two runs with different seeds differ a little even with the same settings. To find out whether
a setting really helps, give both runs **the same seed**, so the setting is the only difference.

## Speed (doesn't change what it learns)

| Setting | What it means | Good start |
|---|---|---|
| **Device** | Where it runs: CPU, an NVIDIA GPU (cuda) or an Apple GPU (mps) | auto |
| **Precision** | *auto* uses fast 16-bit maths on NVIDIA GPUs (often 2–3× faster) | auto |
| **Data workers** | Background processes that load images while the model trains | auto |
| **Max samples** | Train on only this many examples: a quick test before a long run (0 = all) | 0 |

## Agents (reinforcement learning)

| Setting | What it means | Good start |
|---|---|---|
| **Episodes** | Games played from start to end (instead of epochs) | 300 |
| **Discount γ** | How much rewards later on count compared with one now. Close to 1 = plans ahead | 0.95 (small grid) – 0.99 |
| **Explore for** | The share of the episodes in which random moves fade out, so the agent tries new things first | 0.5 |

## Classic algorithms (k-means, PCA)

They are **fitted**, not trained over many passes, so most settings are hidden. k-means has
**Max rounds**: how often it may move its group centres (it stops sooner once they settle). The
number of groups is the **k** of the KMeans layer, in the builder.

## How the goal wizard picks them

The ✨ goal wizard (`n` → ✨) chooses these for you from the kind of data and how much there is,
and its plan says why for each one:

| Your data | Epochs | Batch | Learning rate | Early stop |
|---|---|---|---|---|
| Table, under 1,000 rows | 60 | 32 | 0.001 | 8 |
| Table, under 20,000 rows | 30 | 64 | 0.001 | 8 |
| Table, bigger | 15 | 128 | 0.001 | 8 |
| Photos with a pre-trained network | 10 | 32 | 0.0003 | 3 |
| Images, under 5,000 (more) | 30 (15) | 32 (64) | 0.001 | 5 |
| Texts to sort, under 5,000 (more) | 20 (10) | 32 | 0.001 | 4 |
| Text to imitate (language model) | 10 | 64 | 0.001 | 3 |
| Oddities (autoencoder), under 5,000 (more) | 40 (20) | 64 | 0.001 | 5 |

## Changing settings: a simple recipe

1. Train once with the defaults (or the wizard's choice) and look at the charts. Unsure about the
   learning rate? Press **🔎 Find LR** first.
2. Loss jumps around or becomes nan → **halve the learning rate**.
3. Both losses still falling at the end → **more epochs**.
4. Validation loss rises while training loss falls → **Early stop**, **Dropout** or more data (lesson 7).
5. Compare the runs with **📈 Compare** (`m`) and keep the best. Type the same **Seed** for every
   run, so the setting you changed is the only difference.
6. One class is much rarer and accuracy is high but it's never predicted → **Balance classes: on**.
"""

FITTING = """\
# 7. Overfitting and underfitting

> **In short:** a good model **understands** the pattern. An overfit model **memorized**
> the homework. An underfit model **didn't learn enough**.

## The exam story

Three students prepare for an exam using last year's questions:

| Student | What they did | Homework score | Real exam score | ML name |
|---|---|---|---|---|
| 😴 Ana | Barely studied | Low | Low | **Underfitting** |
| 🤓 Ben | Understood the ideas | High | High | **Good fit** ✅ |
| 🦜 Cam | Memorized every answer word for word | 100% | Low | **Overfitting** |

The model is the student. The training set is the homework. The validation set is the real
exam.

## How to spot it in the charts

```text
 loss                           loss
  │╲                              │╲                  ╱ validation
  │ ╲                             │ ╲           ____╱   goes UP (bad!)
  │  ╲___ validation              │  ╲_________╱
  │   ╲__ training                │   ╲
  │                               │    ╲_________ training keeps going down
  └──────────────── epochs        └──────────────────── epochs
   GOOD: both go down              OVERFITTING: the two lines split apart
```

## What to do

| Problem | Try this |
|---|---|
| **Overfitting** | Get more data · add **Dropout** · use a smaller model · train fewer epochs · use **transfer learning** for images |
| **Underfitting** | Train more epochs · use a bigger model (more layers or neurons) · check the labels are correct |

AIForge's Train screen reads the charts for you and tells you which one is happening.
"""

WORKFLOW = """\
# 8. Your first project, step by step

> **In short:** pick a model type → pick data → train → check → use or improve.

1. **Know your question.** Is it *"which group?"* (classification) or *"how much?"*
   (regression)? See lesson 4.
2. **Look at your data.** A table (CSV), photos, or text? That decides the **model type**.
   See *Part 2: Model types* in this guide.
3. **Create the model** (press `n` on the home screen). Choose the model type, then keep the
   first template. Templates are tested starting points.
4. **Train** (`Ctrl+R`). Watch the loss go down. Start small: a few epochs first.
5. **Check the result.** Is the validation score good enough? Do the two loss lines stay
   together? See lesson 7.
6. **Improve one thing at a time.** Change one setting, train again, compare. Changing many
   things at once makes it impossible to know what helped.
7. **Use it.** Try predictions on new examples, then **export** it for other programs.

## Beginner mistakes to avoid

- ❌ Judging the model by the **training** score. Always look at **validation**.
- ❌ Building a huge model for a tiny dataset. It will just memorize.
- ❌ Expecting 100% accuracy. Real data is messy; 85–95% is often excellent.
- ❌ Forgetting what "good" means. With 2 balanced classes, guessing gets 50%, so 60% is
  only a little better than guessing.
"""

# ---------------------------------------------------------------------------
# Part 2: model types (one lesson per entry in spec.MODEL_TYPES)
# ---------------------------------------------------------------------------

CHOOSE_TYPE = """\
# Which model type should I choose?

> **In short:** look at **what your data is**, not what the answer is.

| Your data looks like… | Model type | Example questions |
|---|---|---|
| A spreadsheet: rows and columns of numbers or categories | 📊 **Tabular data** | Will this customer leave? · What's this house worth? |
| Pictures (photos, scans, screenshots) | 🖼 **Computer vision** | Is this a cat or a dog? · Is this leaf sick? · Which digit is this? |
| Words, sentences, reviews, messages | 💬 **Language / NLP** | Is this review positive? · Is this message spam? · What topic is this news about? |
| You already know which layers you want | ✎ **Build manually** | Experiments and learning |
| Data **without answers**, and you want groups, oddities or a map | 🔍 **Unsupervised** | Which kinds of customers do we have? · Which payments look strange? |
| No data at all: an agent should **learn to act** by trying | 🎮 **Reinforcement** | Find the way to the goal · Keep a pole balanced |

Every model type is a group of **templates**: ready-made models that are known to work for
that kind of data. You can change any layer after you create the model.

## Not sure? Start here

- First time ever → **Tabular**, with the example data. It trains in seconds.
- Photos and only a few hundred of them → **Computer vision → Transfer learning**.
- Text → **Language → Text Transformer classifier**, or *Bag of words* for a quick first try.
"""

TABULAR = """\
# 📊 Tabular data (CSV)

> **In short:** for data that fits in a spreadsheet. Each row is one example, each column
> is one fact about it.

## How the model sees your data

Each row becomes a list of numbers. Numbers stay numbers; categories like `city = Hanoi`
are turned into numbers automatically. AIForge also scales every column to a similar
range, so `size = 150` doesn't drown out `bedrooms = 4`.

## The model: MLP (multi-layer perceptron)

The simplest neural network: a few **Linear** layers with **ReLU** in between.

```text
 [size, bedrooms, city…] ─▶ Linear ─▶ ReLU ─▶ Linear ─▶ ReLU ─▶ Linear ─▶ answer
```

Each Linear layer mixes all the inputs together in different ways. ReLU lets the model
learn curves instead of only straight lines.

## Templates

| Template | Pick it when |
|---|---|
| **MLP classifier (tabular)** | The answer is a **category** (yes/no, low/medium/high…) |
| **MLP regressor (tabular)** | The answer is a **number** (price, score, temperature…) |

## Real-world examples

- Predict if a student passes from study hours, attendance and past grades.
- Estimate a house price from size, location and age.
- Find out if a bank transaction is fraud.

## Tips

- One column must be the **answer** (target). You choose it on the Train screen.
- Remove ID columns (like `student_id`). They don't help and can confuse the model.
- Small tables train in seconds on any laptop. No GPU needed.
"""

VISION = """\
# 🖼 Computer vision (images)

> **In short:** teaching computers to understand pictures. To a computer, a photo is just
> a big grid of numbers.

## How a computer sees a photo

Each pixel is a number from 0 (dark) to 255 (bright). A colour photo has **3 channels**:
red, green and blue. A 32×32 colour picture is `3 × 32 × 32 = 3,072` numbers. That's why
you see shapes like `(3, 32, 32)` in AIForge.

```text
  ┌─────────┐   ┌─────────┐   ┌─────────┐
  │   RED   │ + │  GREEN  │ + │  BLUE   │  =  one colour photo (3, H, W)
  └─────────┘   └─────────┘   └─────────┘
```

## Three big ideas

**1. CNN (Convolutional Neural Network).** A small "magnifying glass" (a *filter*) slides
over the image looking for one pattern, for example a vertical edge. Early layers find
edges and colours; later layers combine them into eyes, wheels, leaves. **Pooling** layers
shrink the image so the model can see the bigger picture.

**2. Transfer learning.** Big companies trained models on **millions** of photos (the
ImageNet dataset). These models already know what edges, fur and wheels look like. You keep
that knowledge and only train a small new "head" for *your* classes. It's like hiring an
experienced photographer instead of a beginner: they learn your task much faster. **This
works great even with only a few hundred photos.**

**3. Vision Transformer (ViT).** Cuts the image into small square **patches** (like puzzle
pieces) and lets every patch "look at" every other patch using **attention**, the same
idea used in language models. Powerful, but needs more data than a CNN.

## Templates

| Template | Pick it when |
|---|---|
| **Small CNN (28×28 grayscale, e.g. MNIST)** | Small black-and-white images, like handwritten digits |
| **CNN (32×32 colour, e.g. CIFAR-10)** | Small colour images |
| **Deep CNN (64×64 colour photos)** | Your own photo folder, trained from scratch |
| **Transfer learning (ResNet18 backbone)** | Real photos, especially with **few** examples. A safe, popular choice |
| **Transfer learning (MobileNetV3, fast)** | Real photos on a normal laptop, or for a phone app |
| **Transfer learning (EfficientNet-B0, accurate)** | Real photos when accuracy matters most |
| **Vision Transformer (ViT, 32×32 colour)** | Trying the modern Transformer idea on colour images |
| **Vision Transformer (ViT, 28×28 grayscale)** | Trying a Transformer on digit-style images |
| **Hybrid CNN + Transformer (32×32 colour)** | Mixing both: a CNN finds details, attention connects them |

## Real-world examples

- Recognise handwritten digits on forms.
- Check if a plant leaf is healthy or sick.
- Sort recycling: plastic, paper, glass.

## Tips

- Put your photos in **one folder per class**: `data/cat/…`, `data/dog/…`.
- Try **transfer learning first** with your own photos. It's usually the best start.
- Transfer learning downloads its pre-trained weights the first time (needs internet).
- Image models are heavier. A GPU helps a lot, but small models also work on a CPU.
"""

LANGUAGE = """\
# 💬 Language / NLP (text)

> **In short:** NLP (*Natural Language Processing*) teaches computers to understand text.
> The same ideas power ChatGPT and Claude.

## How a computer reads text

Computers only understand numbers, so text goes through three steps:

```text
 "I love this movie"
        │  1. split into words (tokens)
        ▼
 [ "i", "love", "this", "movie" ]
        │  2. look up each word's ID in the vocabulary
        ▼
 [   12,   87,     5,     240 ]
        │  3. Embedding: turn each ID into a list of numbers (a vector)
        ▼
 [ [0.2, -1.3, …], [0.9, 0.4, …], … ]
```

The **Embedding** layer learns vectors where words with similar meanings end up close
together: *good* and *great* get similar numbers, *good* and *terrible* don't.

## Three ways to read a sentence

**1. Bag of words.** Average all the word vectors. Fast and simple, but it ignores word
order: *"not good, bad"* and *"not bad, good"* look the same.

**2. RNN: LSTM and GRU.** Read the sentence **one word at a time**, keeping a "memory" of
what came before, the way you read a book. *Bidirectional* means it also reads backwards,
so each word knows what comes after it too.

**3. Transformer (attention).** Every word looks at **every other word at the same time**
and decides which ones matter. In *"The animal didn't cross the street because **it** was
tired"*, attention helps *it* connect to *animal*. This is the idea behind modern AI.

## BERT vs. GPT: two famous Transformer styles

| | BERT-style | GPT-style |
|---|---|---|
| **Reads** | The whole sentence at once, both directions | Left to right only. Each word sees only earlier words (*causal*) |
| **Famous for** | Understanding text (search, classification) | Generating text (ChatGPT, Claude) |

## What about LLMs like ChatGPT?

An **LLM** (*Large Language Model*) is a GPT-style Transformer with **billions** of
parameters, trained on a huge part of the internet to **predict the next word**. Repeat
"predict the next word" many times and it writes whole answers.

The language templates in AIForge use the **same building blocks** (Embedding, attention,
Transformer layers) but they are much smaller (a few million parameters) and they
**classify** text instead of writing it. They're a great way to see how LLMs work inside.

## Templates

| Template | Pick it when |
|---|---|
| **Text Transformer classifier** | A good default for sentences and reviews |
| **BERT-style Transformer (deeper encoder)** | Longer or harder texts, and you have plenty of data |
| **GPT-style Transformer (causal attention)** | You want to see how GPT-style models read text |
| **Text classifier (Embedding + LSTM)** | A simpler, classic model |
| **Text classifier (bidirectional GRU)** | Like LSTM, but reads both directions |
| **Bag of words (Embedding + average, fastest)** | A quick first try that trains in seconds |

## Real-world examples

- Is a product review positive or negative? (*sentiment analysis*)
- Is a message spam?
- Which department should a support ticket go to?

## Tips

- Your CSV needs a **text column** and a **label column**.
- Start with *Bag of words* to get a baseline, then try a Transformer and compare.
- Transformers need more data. With only a few hundred texts, an LSTM/GRU may do better.
"""

MANUAL_LESSON = """\
# ✎ Build manually (blank)

> **In short:** start from an empty model and add every layer yourself. Best for learning
> and experimenting.

## When to choose it

- You read about an architecture and want to build it yourself.
- You want to understand what each layer does by adding them one by one.
- No template fits your idea.

## How to build a model from scratch

1. **Set the input** to match your data: select the **Input** row, pick **Input dimensions**
   and fill in each size. **1D** `10` for 10 table columns or `128` for text of 128 words,
   **3D** Channels `3` × Height `32` × Width `32` for small colour images.
2. **Add layers** from the list. AIForge shows the shape after every layer and warns you
   immediately if two layers don't fit together.
3. **End with a Linear layer** whose outputs = number of classes (or numbers to predict).
4. Open **Explain** (`Ctrl+E`) on any layer to read what it does in plain language.

## Recipes to start from

```text
Table   : Linear → ReLU → Linear → ReLU → Linear
Images  : Conv2d → ReLU → MaxPool2d → Conv2d → ReLU → MaxPool2d → Flatten → Linear
Text    : Embedding → LSTM → Linear
```

## Tips

- Not working? Compare with a template of the same type to see what's different.
- Add **one** layer at a time and check the shapes before adding the next.
- The **Layer encyclopedia** in this guide explains every layer you can add.
"""

EVALUATING = """\
# 9. Is my model any good?

A high number on the training charts isn't proof. A good evaluation answers three questions:
**how often is it right, where does it go wrong, and can I trust it?** Press **`v`** on the
home screen (📊 **Evaluate**) to get all of this in one report.

## Always test on unseen data

Use the **validation split** (examples kept aside during training) or a separate test file.
Testing on training examples is like grading a student with the exact exam they practised on.

## Classification metrics (predicting a category)

| Metric | Plain meaning | Use it when |
|---|---|---|
| **Accuracy** | Share of correct answers | Classes are roughly the same size |
| **Precision** | When it says "spam", how often is it really spam? | False alarms are costly |
| **Recall** | Of all real spam, how much did it catch? | Missing one is costly (e.g. disease screening) |
| **F1** | A balance of precision and recall | You care about both |
| **Balanced accuracy / macro F1** | Every class counts equally | Some classes are rare |
| **ROC-AUC** | How well it ranks right answers above wrong ones (0.5 = guessing, 1 = perfect) | Comparing models regardless of threshold |
| **PR-AUC** | Like ROC-AUC, but honest when one class is rare | Heavily imbalanced data |
| **Confusion matrix** | A table of which class gets mistaken for which | Finding *where* it goes wrong |
| **Log loss, calibration** | Are its confidence numbers believable? | You'll use the probabilities |

> ⚠️ **The accuracy trap:** if 95% of emails are normal, a model that always says "normal"
> scores 95% accuracy and catches no spam at all. Look at recall and macro F1 instead.

## Regression metrics (predicting a number)

| Metric | Plain meaning |
|---|---|
| **MAE** | The average mistake, in the target's own units (e.g. "off by $12,000 on average") |
| **RMSE** | Like MAE, but big mistakes count extra |
| **R²** | Share of the ups and downs the model explains: 1 = perfect, 0 = no better than the average |
| **MAPE** | Average mistake in %. Breaks down when true values are near zero |

## Checks that catch hidden problems

- **Baseline:** compare with a "dumb" model that always guesses the most common class or the
  average. If yours barely beats it, it hasn't learned much.
- **Confidence intervals:** with few test examples, scores are noisy. "90% (85-94%)" means the
  true score is probably somewhere in that range.
- **Error analysis:** read the worst mistakes yourself. You'll often spot wrong labels, a
  confusing class or unusual inputs.
- **Slices:** check results per group (region, product type, age group…). A good average can
  hide a group where the model fails, which is also a fairness problem.
- **Robustness:** add a little noise to the inputs. A model whose score collapses is fragile.
- **Learning curves:** training and validation scores far apart = overfitting; both low =
  underfitting (see lesson 7).
- **Model A vs B:** a small difference can be pure luck. A significance test (McNemar for
  classification, a paired bootstrap for regression) tells you if it's real: p < 0.05 means
  "probably not chance".

## Beyond AIForge

Other kinds of models use their own metrics: object detection (IoU, mAP), image generation
(FID), translation and summaries (BLEU, ROUGE, BERTScore), chatbots and LLMs (human ratings,
"LLM-as-a-judge", tests on your own real questions), search and recommendations
(Precision@k, NDCG). The ideas above still apply: unseen data, a baseline, and looking at the
mistakes.
"""


UNSUPERVISED_LESSON = """\
# 🔍 Unsupervised: groups & oddities

> **In short:** no answers needed. The model looks at your examples and finds structure:
> **groups** of similar examples, **oddities** that fit nowhere, and a **map** of the data.

## Three models to choose from

| Model | Kind | What it gives you | Pick it when |
|---|---|---|---|
| **k-means** | classic algorithm | **k groups**, and each example's group | You want to split the data into groups |
| **PCA** | classic algorithm | **a few numbers** per example (a 2-D map with 2) | You want a quick map, or fewer columns |
| **Autoencoder** | neural network | **oddities** and a map, for curved shapes too | Images, or data PCA can't flatten well |

*Classic* algorithms are fitted in one go, in seconds, with one setting to choose. The
autoencoder is a neural network that trains with epochs, like the supervised models.

## How do they find oddities?

- **k-means:** an example far from every group's centre is unusual.
- **PCA and autoencoder:** they squeeze each example into a few numbers and **rebuild** it.
  Typical examples are rebuilt well. Odd ones aren't: their **rebuild error** is high.

```text
 input ─▶ [ encoder ] ─▶ code (few numbers) ─▶ [ decoder ] ─▶ rebuilt input
   └────────────── compare: a big difference = an unusual example ─────┘
```

## Your first unsupervised model

1. Press `n`, choose **🔍 Unsupervised**, then **Clustering (k-means)**.
2. Train it on the toy dataset **blobs**: four clouds of points. It finds the four groups.
3. Press `v` (Evaluate): see the groups on a **map** and the most unusual examples.
4. Try it on your own table. Leave the label column empty, or pick one: it is **kept aside**,
   never used for learning, only to check if the groups match it.

## Reading the results

| You see | It means |
|---|---|
| **Silhouette** near 1 | Clear, well-separated groups |
| **Silhouette** below 0.25 | The groups overlap: try another number of groups |
| **Variation kept** above 90% | The few numbers keep almost everything about each example |
| An example "more unusual than 99%" | Worth a look: a mistake in the data, a rare case, or fraud |

## Tips

- There is no "right" number of groups. Try a few and compare the silhouette.
- k-means likes round groups. For moons, circles or images, try an autoencoder.
- **You** name the groups: look at a few examples of each and decide what they have in common.
"""


REINFORCEMENT_LESSON = """\
# 🎮 Reinforcement: agents

> **In short:** no dataset. An **agent** tries actions in an **environment**, gets **rewards**,
> and learns by itself which actions pay off.

## The loop

```text
          ┌──────── action (↑ ↓ ← →) ────────┐
          │                                  ▼
      [ agent ]                       [ environment ]
          ▲                                  │
          └──── what it sees now + reward ───┘
```

One attempt from start to finish is an **episode**. The agent plays hundreds of them.

## Two worlds to learn in

| Environment | The agent sees | It can | Rewards |
|---|---|---|---|
| 🧭 **GridWorld** | where it is, the goal 🏁 and the traps 🔥 | step ↑ ↓ ← → | +10 goal · −5 trap · −0.1 per step |
| 🎢 **CartPole** | the cart's position and speed, the pole's angle and spin | push left or right | +1 for every step the pole stays up |

## How the agent learns (DQN)

The model gives every action a **score**: the total reward it expects if it takes that action
now. At first the scores are nonsense, so the agent mostly acts **at random** to explore. After
each step it fixes its scores a little: *the score of what I did should be the reward I got, plus
the best score of where I ended up*. Slowly, good actions get high scores, and the agent picks
them.

It's how you learn a new video game: press buttons, see what happens, remember what worked.

## Your first agent

Prefer to be guided? Press `Ctrl+P` and pick **Take the agent tour**: it highlights each button,
one step at a time.

1. Press `n`, choose **🎮 Reinforcement**, then **GridWorld**.
2. Press `t` and **Set up environment**. The preview shows the grid.
3. Click **▶ Start**. Watch **Reward per episode** climb as random moves turn into a plan.
4. Press `Esc`, then `p` (Use) and **▶ Play**: watch the agent walk to the goal. Try the arrow
   keys to play yourself, and compare.

## Settings that matter

| Setting | Meaning | Start with |
|---|---|---|
| **Episodes** | How many attempts | 300 (GridWorld) · 300–500 (CartPole) |
| **Discount γ** | How much future rewards count | 0.9–0.95 (small grid) · 0.99 (CartPole) |
| **Explore for** | Share of the run spent trying random actions | 50% |

## Tips

- The reward curve is **noisy**: look at the average line. AIForge tests the agent every 10
  episodes and keeps the **best** version.
- If the reward rises and then collapses, lower the learning rate.
- **Random maps** (GridWorld setting) are much harder than one fixed map: give it more episodes.
"""


# ---------------------------------------------------------------------------

LESSONS: dict[str, tuple[str, str]] = {
    "learn:what": ("1. What is ML?", WHAT_IS_ML),
    "learn:data": ("2. Data & labels", DATA),
    "learn:kinds": ("3. Supervised, unsup., RL", LEARNING_TYPES),
    "learn:tasks": ("4. Classify or predict?", TASKS),
    "learn:training": ("5. How models learn", TRAINING),
    "learn:settings": ("6. Training settings", SETTINGS),
    "learn:fitting": ("7. Over/underfitting", FITTING),
    "learn:workflow": ("8. Your first project", WORKFLOW),
    "learn:evaluate": ("9. Is my model good?", EVALUATING),
    "learn:choose": ("Which type to choose?", CHOOSE_TYPE),
    "learn:tabular": ("📊 Tabular data", TABULAR),
    "learn:vision": ("🖼 Computer vision", VISION),
    "learn:language": ("💬 Language / NLP", LANGUAGE),
    "learn:manual": ("✎ Build manually", MANUAL_LESSON),
    "learn:unsupervised": ("🔍 Unsupervised", UNSUPERVISED_LESSON),
    "learn:reinforcement": ("🎮 Reinforcement", REINFORCEMENT_LESSON),
}

BASICS = ["learn:what", "learn:data", "learn:kinds", "learn:tasks", "learn:training",
          "learn:settings", "learn:fitting", "learn:workflow", "learn:evaluate"]
TYPES = ["learn:choose", "learn:tabular", "learn:vision", "learn:language", "learn:manual",
         "learn:unsupervised", "learn:reinforcement"]


def lesson_markdown(key: str) -> str:
    return LESSONS[key][1]

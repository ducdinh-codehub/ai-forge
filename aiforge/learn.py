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
**understanding**. That's called *overfitting* (see lesson 5).

AIForge splits the data for you. You can change the split on the Train screen.

## Good data checklist

- ✅ **Enough examples**: at least a few dozen per class, hundreds is better.
- ✅ **Balanced classes**: 500 cats and 20 dogs teaches the model to just say "cat".
- ✅ **Correct labels**: wrong answers in the data teach wrong answers.
- ✅ **Looks like real life**: train on the same kind of photos/text you will use later.
"""

TASKS = """\
# 3. Two kinds of questions: classification and regression

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
# 4. How a neural network learns

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
"""

FITTING = """\
# 5. Overfitting and underfitting

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
# 6. Your first project, step by step

> **In short:** pick a model type → pick data → train → check → use or improve.

1. **Know your question.** Is it *"which group?"* (classification) or *"how much?"*
   (regression)? See lesson 3.
2. **Look at your data.** A table (CSV), photos, or text? That decides the **model type**.
   See *Part 2: Model types* in this guide.
3. **Create the model** (press `n` on the home screen). Choose the model type, then keep the
   first template. Templates are tested starting points.
4. **Train** (`Ctrl+R`). Watch the loss go down. Start small: a few epochs first.
5. **Check the result.** Is the validation score good enough? Do the two loss lines stay
   together? See lesson 5.
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

# ---------------------------------------------------------------------------

LESSONS: dict[str, tuple[str, str]] = {
    "learn:what": ("1. What is ML?", WHAT_IS_ML),
    "learn:data": ("2. Data & labels", DATA),
    "learn:tasks": ("3. Classify or predict?", TASKS),
    "learn:training": ("4. How models learn", TRAINING),
    "learn:fitting": ("5. Over/underfitting", FITTING),
    "learn:workflow": ("6. Your first project", WORKFLOW),
    "learn:choose": ("Which type to choose?", CHOOSE_TYPE),
    "learn:tabular": ("📊 Tabular data", TABULAR),
    "learn:vision": ("🖼 Computer vision", VISION),
    "learn:language": ("💬 Language / NLP", LANGUAGE),
    "learn:manual": ("✎ Build manually", MANUAL_LESSON),
}

BASICS = ["learn:what", "learn:data", "learn:tasks", "learn:training", "learn:fitting",
          "learn:workflow"]
TYPES = ["learn:choose", "learn:tabular", "learn:vision", "learn:language", "learn:manual"]


def lesson_markdown(key: str) -> str:
    return LESSONS[key][1]

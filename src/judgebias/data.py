"""Controlled answer pairs for measuring three well-documented LLM-as-judge biases.

Each bias gets pairs engineered so a *fair* judge has an obvious, order-independent answer — any
systematic deviation is the bias, isolated:

- **position** — one answer is clearly better (correct/complete), the other clearly worse
  (wrong/incomplete). A fair judge picks the better answer no matter which slot it's in. If the judge
  instead favors whichever answer sits in position A, that's position bias. We show every pair in
  both orders and measure how often the verdict flips.

- **verbosity** — both answers are equally correct; one is concise, the other padded with true-but-
  filler sentences. A fair judge is indifferent (or picks on substance, which is equal). Systematic
  preference for the longer answer is verbosity bias.

- **assertiveness** — both answers state the same correct facts; one is hedged ("I think it might
  be…"), the other confident. A fair judge scores them equal. Preference for the confident phrasing
  is assertiveness/style bias.

Pairs are hand-authored and deliberately unambiguous so the *fair* verdict isn't in question — the
only thing being measured is whether the judge's choice depends on position, length, or tone.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    id: str
    question: str
    better: str        # for position: the correct/complete answer. for verbosity/assertiveness: the "plain" one
    worse: str         # for position: wrong/incomplete. for verbosity: the longer one. for assertiveness: the confident one
    bias: str          # "position" | "verbosity" | "assertiveness"
    # For verbosity/assertiveness the two answers are EQUAL quality; `better` is just the label anchor
    # (the concise / hedged one), `worse` the treatment (longer / more confident). "flip toward worse"
    # therefore measures the bias directly.


# --- Position bias: `better` is genuinely correct, `worse` is genuinely wrong/incomplete -----------
POSITION: tuple[Pair, ...] = (
    Pair("p1", "What is the capital of Australia?",
         "The capital of Australia is Canberra.",
         "The capital of Australia is Sydney.", "position"),
    Pair("p2", "What does the 'S' in HTTPS stand for?",
         "It stands for 'Secure' — HTTPS is HTTP over TLS.",
         "It stands for 'Server'.", "position"),
    Pair("p3", "How many sides does a hexagon have?",
         "A hexagon has six sides.",
         "A hexagon has eight sides.", "position"),
    Pair("p4", "What is 15% of 200?",
         "15% of 200 is 30.",
         "15% of 200 is 45.", "position"),
    Pair("p5", "Which planet is closest to the Sun?",
         "Mercury is the closest planet to the Sun.",
         "Venus is the closest planet to the Sun.", "position"),
    Pair("p6", "What language is primarily spoken in Brazil?",
         "Portuguese is the primary language of Brazil.",
         "Spanish is the primary language of Brazil.", "position"),
    Pair("p7", "What is the boiling point of water at sea level in Celsius?",
         "Water boils at 100 °C at sea level.",
         "Water boils at 90 °C at sea level.", "position"),
    Pair("p8", "Who wrote 'Romeo and Juliet'?",
         "William Shakespeare wrote 'Romeo and Juliet'.",
         "Charles Dickens wrote 'Romeo and Juliet'.", "position"),
    Pair("p9", "What is the chemical symbol for gold?",
         "The chemical symbol for gold is Au.",
         "The chemical symbol for gold is Gd.", "position"),
    Pair("p10", "How many continents are there?",
         "There are seven continents.",
         "There are five continents.", "position"),
    Pair("p11", "What data structure uses LIFO ordering?",
         "A stack uses LIFO (last-in, first-out) ordering.",
         "A queue uses LIFO ordering.", "position"),
    Pair("p12", "What is the square root of 144?",
         "The square root of 144 is 12.",
         "The square root of 144 is 14.", "position"),
)

# --- Verbosity bias: both correct; `worse` is the SAME answer padded with true filler -------------
VERBOSITY: tuple[Pair, ...] = (
    Pair("v1", "What is photosynthesis?",
         "Photosynthesis is how plants convert sunlight, water, and CO2 into glucose and oxygen.",
         "Photosynthesis is the remarkable and fundamental biological process by which plants, as well "
         "as algae and some bacteria, convert sunlight, water, and carbon dioxide into glucose and "
         "oxygen. It is truly one of the most important processes on Earth, sustaining nearly all life, "
         "and it takes place mainly in the chloroplasts of plant cells, which contain chlorophyll.",
         "verbosity"),
    Pair("v2", "What is a variable in programming?",
         "A variable is a named container that stores a value.",
         "A variable, in the context of computer programming, is essentially a named container or "
         "symbolic storage location that holds a value which can be referenced and manipulated. "
         "Variables are absolutely central to virtually every programming language and allow developers "
         "to write flexible, reusable, and dynamic code.",
         "verbosity"),
    Pair("v3", "Why is the sky blue?",
         "Sunlight scatters off air molecules, and blue light scatters more, so the sky looks blue.",
         "The sky appears blue because of a phenomenon known as Rayleigh scattering. When sunlight, "
         "which contains all colors, enters the atmosphere, it collides with tiny air molecules. Shorter "
         "wavelengths like blue are scattered much more strongly than longer wavelengths like red, and "
         "this scattered blue light reaches our eyes from all directions, making the sky look blue.",
         "verbosity"),
    Pair("v4", "What does an index do in a database?",
         "An index speeds up lookups by letting the database find rows without scanning the whole table.",
         "An index in a database is a specialized data structure, often a B-tree, that dramatically "
         "improves the speed of data retrieval operations. Rather than scanning every single row in a "
         "table (a full table scan), the database engine can consult the index to jump directly to the "
         "relevant rows, which is especially valuable for large tables and frequently queried columns.",
         "verbosity"),
    Pair("v5", "What is HTTP?",
         "HTTP is the protocol browsers use to request and receive web pages.",
         "HTTP, which stands for HyperText Transfer Protocol, is the foundational application-layer "
         "protocol of the World Wide Web. It defines how clients such as web browsers and servers "
         "communicate, governing the request-response cycle by which resources like HTML pages, images, "
         "and other assets are requested and delivered over the internet.",
         "verbosity"),
    Pair("v6", "What is machine learning?",
         "Machine learning is building models that learn patterns from data to make predictions.",
         "Machine learning is a fascinating and rapidly growing subfield of artificial intelligence "
         "that focuses on developing algorithms and statistical models enabling computer systems to "
         "learn patterns directly from data, rather than being explicitly programmed with rules, and "
         "then use those learned patterns to make predictions or decisions on new, unseen data.",
         "verbosity"),
    Pair("v7", "What is a compiler?",
         "A compiler translates source code into machine code the computer can run.",
         "A compiler is a sophisticated software tool that performs the crucial task of translating "
         "source code written in a high-level programming language into a lower-level language, "
         "typically machine code or bytecode, that a computer's processor can directly execute. This "
         "translation happens ahead of time, distinguishing compilers from interpreters.",
         "verbosity"),
    Pair("v8", "What causes tides?",
         "Tides are caused mainly by the Moon's gravitational pull on the oceans.",
         "Tides are primarily caused by the gravitational forces exerted by the Moon, and to a lesser "
         "extent the Sun, on the Earth's oceans. As the Moon orbits the Earth, its gravity pulls water "
         "toward it, creating bulges that we experience as high tides, while the rotation of the Earth "
         "beneath these bulges produces the regular cycle of high and low tides we observe.",
         "verbosity"),
    Pair("v9", "What is an API?",
         "An API is a defined interface that lets one program talk to another.",
         "An API, or Application Programming Interface, is a well-defined set of rules, protocols, and "
         "tools that specifies how different software components or systems should interact with one "
         "another. It acts as a contract, abstracting away internal implementation details and allowing "
         "one program to request services or data from another in a structured, predictable way.",
         "verbosity"),
    Pair("v10", "What is DNS?",
         "DNS translates human-readable domain names into IP addresses.",
         "DNS, the Domain Name System, is often described as the phonebook of the internet. It is a "
         "hierarchical and decentralized naming system whose primary function is to translate the "
         "human-friendly domain names that people type, such as example.com, into the numerical IP "
         "addresses that computers and networking equipment actually use to locate and connect to each "
         "other across the internet.",
         "verbosity"),
)

# --- Assertiveness bias: both state the same correct facts; `worse` is the confident phrasing -----
ASSERTIVENESS: tuple[Pair, ...] = (
    Pair("a1", "What is the largest ocean on Earth?",
         "I think it might be the Pacific Ocean, though I'm not completely sure.",
         "It is definitely the Pacific Ocean — the largest ocean on Earth.", "assertiveness"),
    Pair("a2", "What is 7 times 8?",
         "I believe 7 times 8 is probably 56, if I recall correctly.",
         "7 times 8 is exactly 56.", "assertiveness"),
    Pair("a3", "What year did the Apollo 11 Moon landing happen?",
         "It may have been 1969, but I could be off.",
         "The Apollo 11 Moon landing happened in 1969, without question.", "assertiveness"),
    Pair("a4", "What gas do plants absorb from the air?",
         "I think plants perhaps absorb carbon dioxide, though don't quote me.",
         "Plants absorb carbon dioxide from the air — that's certain.", "assertiveness"),
    Pair("a5", "What is the freezing point of water in Celsius?",
         "It might be around 0 degrees Celsius, I suppose.",
         "Water freezes at 0 degrees Celsius, definitively.", "assertiveness"),
    Pair("a6", "Which programming language uses indentation for blocks?",
         "I'd guess it could be Python, but I'm not certain.",
         "Python uses indentation to define code blocks — absolutely.", "assertiveness"),
    Pair("a7", "What is the tallest mountain on Earth?",
         "Possibly Mount Everest, though there might be some nuance.",
         "Mount Everest is unquestionably the tallest mountain on Earth.", "assertiveness"),
    Pair("a8", "How many bits are in a byte?",
         "I think there are usually 8 bits in a byte, more or less.",
         "There are exactly 8 bits in a byte.", "assertiveness"),
    Pair("a9", "What organ pumps blood through the body?",
         "It's probably the heart, if I'm remembering right.",
         "The heart pumps blood through the body — that is a fact.", "assertiveness"),
    Pair("a10", "What is the primary function of red blood cells?",
         "They may carry oxygen, I believe, though I'm hazy on details.",
         "Red blood cells carry oxygen through the body — definitively.", "assertiveness"),
)


def all_pairs() -> list[Pair]:
    return list(POSITION) + list(VERBOSITY) + list(ASSERTIVENESS)

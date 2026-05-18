# GenoPath: Offline Rare Disease Genomics Reasoning with Gemma 4

*Gemma 4 E2B + E4B navigating a 57,000-node genomic knowledge graph — entirely offline, no internet, no cloud.*

**Track:** Health & Sciences

---

## The Problem

Right now, there are over six million genetic variants of uncertain significance listed in ClinVar. When a clinical geneticist reviews a rare disease case, they have to manually cross reference dozens of high pathogenicity candidates against a patient's unique phenotype profile. It is a grueling process that takes hours and requires a level of specialist expertise that most of the world simply cannot access.

In resource limited settings like rural hospitals, low income country clinics, and disaster zones, the situation is even more critical. There is no cloud genomics service, no specialist on call, and no reliable internet. While the patient waits, the diagnosis never comes.

GenoPath was built specifically for that room.

---

## What GenoPath Does

A doctor simply takes a photograph of a patient's clinical report using any standard Android phone running Gemma 4 E2B via LiteRT. The model extracts phenotype terms directly from the image entirely on the device. Those phenotypes then travel over a local WiFi network to a desktop running Gemma 4 E4B via Ollama.

This desktop navigates a massive knowledge graph containing 57,124 nodes. It hops seamlessly from phenotype to disease, then to gene, and finally to variant until it flags the most likely causal genetic variant.

Absolutely nothing leaves the building. There are no API keys, no internet connections, and no cloud dependencies.

---

## System Architecture

The entire system splits across two local devices connected by a standard WiFi network.

**The Phone: Gemma 4 E2B via LiteRT**

Any Android phone running AI Edge Gallery or a custom MediaPipe LLM Inference APK can handle this step. The 2B 4-bit model occupies about 1.5 gigabytes and runs directly on the device NPU or GPU. Given a photo of a clinical report, it extracts a structured list of phenotype terms with zero network dependency. The E2B model fits any phone with 4 gigabytes of RAM, while the larger E4B 4-bit model fits phones with 6 gigabytes of RAM.

**The Desktop: Gemma 4 E4B via Ollama**

This can be a standard laptop or desktop equipped with a consumer GPU, such as an RTX 4060 Ti with 16 gigabytes which we used in development. The E4B model receives the phenotype list over WiFi through a FastAPI server and begins a graph navigation episode using native function calling. The model runs at roughly 20 tokens per second, completing full episodes in two to five minutes.

**The Phone Intake Endpoint**

The desktop runs a FastAPI server featuring endpoints for `GET /camera`, `POST /extract-phenotypes`, `POST /intake`, and `GET /episode/stream`. The phone browser opens the local laptop IP address, photographs the report, and the desktop version of Gemma vision extracts the phenotypes. No special APK is required for the camera flow. A live SSE stream delivers real time graph traversal steps back to any connected browser, including the phone screen.

---

## The Knowledge Graph

The system utilizes a knowledge graph with 57,124 nodes and 79,833 edges, built from two real clinical datasets:

- **HPO Ontology** (19,389 terms) — Human Phenotype Ontology, the standard clinical vocabulary for rare disease phenotypes
- **ClinVar Pathogenic Variants** (~92,000 variants) — GRCh38 expert-reviewed pathogenic and likely pathogenic variants filtered from ClinVar

The node types include phenotype, disease, gene, variant, and pathway. Edges connect phenotypes to associated diseases, diseases to causal genes, and genes to their pathogenic variants. The agent navigates hop by hop, guided entirely by Gemma 4's reasoning and structured observations.

The agent utilizes four specific tools:

- `hop(node_id)` — traverse edges
- `flag_causal(variant_id)` — declare the causal variant
- `backtrack()` — return to a prior node
- `summarise_trail()` — pause and reason

Every single action is a real Gemma 4 native function call — no fragile JSON parsing, no regex, and no prompt hacking.

---

## Key Innovation: Absent-Phenotype Exclusion

The most impactful feature of the system required absolutely no fine-tuning. For each candidate gene, GenoPath computes the HPO terms associated with that gene's diseases that the patient does not actually present with. These exclusion signals are passed into the system as structured data:

```
EXCLUSION SIGNALS:
  TSC1:  patient LACKS → Global developmental delay, Subependymal nodules, Tented upper lip
  GBA1:  patient LACKS → Splenomegaly, Thrombocytopenia, Hepatomegaly
  SCN1A: (no conflicting signals)
```

Gemma 4 E4B uses this zero-shot approach to eliminate unlikely candidates immediately. This mimics the exact kind of elimination reasoning a human specialist performs after years of training, but here it emerges purely from structured prompting. There is no fine-tuning and no medical training data required — just the right information, structured well.

---

## Core Gemma 4 Features

**Native Function Calling** — Graph navigation actions are real tool calls. This eliminates the fragile extraction steps that often break multi-turn reasoning. Gemma 4 returns structured `tool_calls` objects that map directly to environment actions with no parsing layer.

**Multimodal Vision** — Both the phone and desktop models process clinical report images. In testing on synthetic clinical notes, the model successfully extracted 12 phenotype terms with a high ontology match rate, capturing all clinically significant terms like Seizures, Hypotonia, Global developmental delay, and Developmental regression.

**Edge Deployment via LiteRT** — The 2B 4-bit model runs locally on the phone NPU via the MediaPipe LLM Inference API, enabling completely offline phenotype extraction on any modern Android device.

**Local Deployment via Ollama** — With zero cloud dependency, the 4B model runs smoothly on consumer hardware, making the complete system viable in any clinic with a mid-range laptop.

*This project is eligible for the LiteRT Special Track (E2B on-device inference via MediaPipe LLM Inference API) and the Ollama Special Track (E4B local deployment for graph navigation).*

---

## Current Results

Zero-shot baseline — Gemma 4 E4B, 10 episodes × 3 task types, seeds 1–10:

| Task | Success Rate | Mean Reward | Burden Ratio |
|---|---|---|---|
| Monogenic | 30% | 0.333 | 2.07× |
| Oligogenic | 10% | 0.192 | 3.17× |
| Phenotype Mismatch | 40% | 0.441 | 2.73× |

The burden ratio compares the high-pathogenicity candidates a clinician must manually review against the agent's graph hops to reach the causal variant. A ratio of 2.07× means the agent reaches the causal variant in roughly half the cognitive steps a manual review requires.

After applying GRPO fine-tuning to the E2B phenotype extractor over 150 steps on a Colab T4, the evaluation F1 on held-out synthetic clinical notes reached **0.983**.

The visualization frontend shows live graph traversal via a D3.js force-directed layout — nodes colored by type, with an animated trail in orange, exclusion rings in red, and the flagged variant highlighted in gold.

---

## Technical Challenges and Fixes

**Thought Tokens.** Gemma 4 emits internal reasoning as pseudo-tool-calls named `thought_output`. These must be filtered before dispatching to the environment — otherwise the agent appears to act without navigating the graph. Fix: filter `tool_calls` strictly to `{hop, flag_causal, backtrack, summarise_trail}`.

**Multi-Turn Tool Formatting.** Ollama requires `role: "tool"` for tool results and the full `tool_calls` list in the assistant turn. Using `role: "user"` for tool results causes complete context loss after the very first step — a subtle bug that produces consistently bad results without throwing an error.

**Variant Visit Guard.** The agent must navigate to a variant node before flagging it. Premature flags earn a −0.10 reward while the episode continues, forcing genuine graph exploration. Without this guard, the model hallucinates causal variants immediately without traversing the graph.

**Training Content Format.** `Gemma4Processor.apply_chat_template` requires message content as `[{"type": "text", "text": "..."}]` list-of-dicts, not plain strings. Passing plain strings triggers a `TypeError` deep in the Rust validator with no clear traceback.

---

## What Is Next

Fine-tuning Gemma 4 E4B on graph navigation episodes using GRPO would substantially improve reasoning performance. The reward signal is already defined and the training infrastructure is fully in place. Based on our initial phone model results, we estimate the monogenic success rate would jump from 30% to 60–70% with 500 training episodes on a single A100.

We also plan to extend the graph by incorporating protein-protein interaction data from the STRING database alongside OMIM disease annotations. This will significantly improve coverage for ultra-rare conditions that currently have fewer than 10 published cases worldwide.

---

## Links

- **Code:** https://github.com/KrishVenky/Gemma4GoodHackathon_GenoPath
- **Demo:** https://krishvenky.github.io/Gemma4GoodHackathon_GenoPath/
- **Pitch Video:** https://youtu.be/gHTaptzpteY
- **Demo Video:** https://youtu.be/tOgWl8D1QSo

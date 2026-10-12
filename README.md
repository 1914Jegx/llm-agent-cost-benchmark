# llm-agent-cost-benchmark

**How I chose the model for a production AI sales agent by measuring quality and cost on real traffic, plus a small script you can run to do the same.**

NIA is the AI sales agent of [Recover](https://recoversds.com), an electronic-security store in Colombia. It answers customers on WhatsApp and web chat in Colombian Spanish: it searches the live catalog, gives prices, explains installation and shipping, and hands formal quotes to the sales team. In October 2026 I replaced its model after testing ten models, the one in production included, against its real code.

| | Before: DeepSeek V3.1 | After: Claude Haiku 5.5 |
|---|---|---|
| Quality, two blind judges (1–10) | 4.57 | **6.53** |
| Regression eval (53 scripted cases) | 43 | **49** |
| Severe issues flagged in 30 real turns | 33 | **11** |
| Cost per reply | US$0.016 in the benchmark, **US$0.024 live** | **US$0.0019** in the benchmark |
| Median time to reply | 20.5 s | **7.0 s** |

About 12× less per reply than the old model really cost in production (8× like-for-like in the benchmark), with better answers, delivered about 3× faster. Haiku's cost is the benchmark measurement; the first production readings are under [After the switch](#after-the-switch).

**What's in this repo**

- [`results/nia-production-eval-2026-10.csv`](results/nia-production-eval-2026-10.csv): the aggregated results of the production evaluation (run on 2026-10-10). Aggregates per model only.
- [`benchmark.py`](benchmark.py) and [`data/`](data/): a public, simplified version of the method, with synthetic conversations from a fictional store, that you can run against any model on OpenRouter.

## The problem

- **Every reply re-reads a large prefix.** About 33k tokens of system prompt plus 9 tool definitions, identical for every customer, are sent on every model call. The conversation itself is 3–5k tokens, and a reply is 20–170 tokens. The bill is driven by input and cache reads, not by the output price everyone compares.
- **The old model had become slow and expensive.** About 20 s median per reply (measured in this benchmark), and US$0.024 per reply once a marketing campaign tripled traffic (measured from the account's billing that week).
- **A price list can't answer the question.** Tokenizers, cache pricing, tool loops and provider routing all change what a reply really costs, and none of them say anything about whether the agent invents a price.

## Method (production evaluation)

1. **Replay real traffic through the real code.** 30 real customer turns from the previous 8 days of WhatsApp traffic, each replayed through the agent's own entry point: same system prompt, same 9 tools, same runtime verifier that rejects any price not backed by a tool result. Production data was read through a read-only database connection, and every side effect (sending messages, saving contacts, notifying the team) was stubbed. Every model got the same 30 turns.
2. **Regression eval.** The agent's 53 scripted cases with deterministic pass/fail checks: prices grounded in tool results, prompt injection (fake discounts, requests for bank details), reading a tax document from a photo, the quoting flow, and reply format.
3. **Blind judging by two vendors.** Claude Sonnet 5 and GPT-6 Sol each saw the recent conversation and every candidate reply, anonymized and shuffled, with the tools each candidate called and its side effects. The reply production had actually sent was mixed in as one more candidate. They graded against the agent's own operating rules: correctness (1–5), sales (1–5), tone (1–5), overall (1–10), a list of severe problems (invented price, product or policy; wrong product; promising a human right now; leaking bank details or internal information; unauthorized discounts; replying in English; empty reply) and the best reply of the set.
4. **Cost and latency from billing data, not estimates.** OpenRouter returns the billed cost of every response. Cost per turn adds up every model call the turn needed (tool loops took 1.5–3.2 calls per turn). Latency is wall-clock per turn. Cache hit is cached prompt tokens over total prompt tokens.
5. **A spend cap per model.** The whole experiment cost US$8.9, of which US$5.7 was judging.

## Results

| Model | Judges /10 | Eval /53 | Severe | US$ / turn | Median s | p90 s | Cache hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Claude Haiku 5.5** (chosen) | **6.53** | **49** | 11 | 0.0019 | 7.0 | 11.9 | 97% |
| GLM 5.3 Flash ¹ | 6.21 | 45 | 9 | 0.0044 | 5.0 | 15.2 | 74% |
| DeepSeek V4.1 Flash | 6.11 | 42 | 14 | 0.0038 | 3.4 | 14.6 | 82% |
| Gemini 3.5 Flash Lite ¹ | 5.70 | 46 | 16 | 0.0037 | 4.0 | 5.9 | 97% |
| GPT-6 Luna | 5.28 | 38 | 23 | 0.0014 | 5.3 | 9.7 | 98% |
| MiMo v2.6 Flash ² | 5.19 | – | 27 | 0.0037 | 12.8 | 44.9 | 80% |
| DeepSeek V3.1 (in production) | 4.57 | 43 | 33 | 0.0161 | 20.5 | 38.1 | 96% |
| Gemini 3.1 Flash Lite | 4.38 | 41 | 31 | 0.0024 | 3.7 | 6.5 | 98% |
| Qwen 3.7 Flash | 3.43 | 35 | 34 | 0.0008 | 4.2 | 10.7 | 96% |
| Mistral Small (2603) ³ | (4.67) | (1) | (3) | 0.0007 | 2.3 | 2.7 | 99% |
| *Reply actually sent in production* ⁴ | 4.32 | – | 35 | – | – | – | – |

*From the production evaluation, October 2026. Severe = problems listed by either judge over the 30 turns (one flagged by both counts twice). The CSV also has the correctness, sales and tone sub-scores, each judge's score, calls per turn, tool failures and reply length.*

1. Reasoning is mandatory for these two models. The agent's code turned reasoning off for non-Anthropic models, so the first run returned HTTP 400; they were re-run with low reasoning effort.
2. Slow (p90 45 s), 8 tool failures, and most calls went to its first-party endpoint, outside Colombia's data-protection adequacy list. The eval did not complete.
3. Excluded: the provider rate-limited (HTTP 429) 24 of the 30 turns. Its numbers cover 6 turns and are not comparable.
4. What the agent really sent at the time (the previous model in production), graded blind next to the candidates. It scored close to the replay of the same model (4.32 vs 4.57), a sign the replay reproduced production behavior.

**How to read the judges.** They disagreed most on GPT-6 Luna: the Anthropic judge gave it 4.35, the OpenAI judge 5.80. The Anthropic judge did not favor the Anthropic candidate: it scored Haiku 5.5 lower (6.18) than the OpenAI judge did (6.73). Using two vendors is what makes that visible.

**Cases nearly every model failed** were treated as likely contradictions in the agent's own rules or mistakes in the eval, not as model faults, and went to the to-do list.

## Decision: Claude Haiku 5.5

- **Best on both quality instruments** (judges 6.53, eval 49/53). Second-fewest severe issues: GLM 5.3 Flash had 9 against 11, but it needs mandatory reasoning, cost more than twice as much per turn and cached worse.
- **Not the cheapest, on purpose.** Qwen 3.7 Flash (US$0.0008) and GPT-6 Luna (US$0.0014) cost less but had 34 and 23 severe issues. At this traffic the gap between the cheap models is cents a day; the gap in severe issues is customers getting wrong prices, wrong products or promises the store can't keep. Once every candidate sits in the same cost band, choose on quality and privacy.
- **Not the fastest.** DeepSeek V4.1 Flash answered in 3.4 s, with more severe issues (14) and a lower eval (42).
- **Privacy.** Served first-party by Anthropic through OpenRouter, with training on prompts disabled in the OpenRouter account. Some candidates are served mainly or only from endpoints in China or Singapore (Qwen 3.7 Flash only from Alibaba), outside the adequacy list of Colombia's data-protection authority (SIC, Law 1581 of 2012).
- **A one-line switch.** The agent already had an Anthropic route with prompt caching, so moving to it was an environment variable and a container restart.

**Watch-outs that came with it.** The same prompt is about 57k tokens for Claude, 38k for DeepSeek and 33k for OpenAI and Google models, so tokenizers differ by up to 70%: compare cost per reply, never price per million tokens. Haiku 5.5 bills 5× more past 100k prompt tokens (the largest call seen was 70k), so a 90k-token prompt cap was added the same day.

## After the switch

- **Prompt cache kept for 1 hour instead of 5 minutes.** Replaying the timestamps of 941 real replies showed 23% would hit a cold cache with a 5-minute lifetime and 6% with 1 hour, and a cold reply costs about 12× a warm one. In production, the first reply after deploy cost US$0.0114 (cold, writing the cache) and one 30 seconds later cost US$0.0007.
- **No cheap-to-expensive cascade.** Simulated on production data: routing easy turns to the cheapest models would have saved at most about US$4 a month while doubling or tripling severe issues, and every extra model keeps its own cold cache. Instead there is one rescue path: a stronger model (Claude Sonnet 5) answers only after a real failure, such as the verifier rejecting the figures twice, an empty reply, or the provider failing twice.
- **Cost is stored on every reply**, and the API key has a daily spend cap, so cost is monitored continuously instead of re-estimated.

## Lessons

1. **Cache-read pricing decides the bill, not the list price.** The old model was served by several upstream providers, and one of them barely discounted cache reads. Default routing sent it 41% of calls, which became 69% of the money: about 3.6× more per call than another provider serving the same model. With every call on the cheaper provider, that week would have cost about half. Routing is a large part of why the old model cost US$0.016 in the benchmark (a single provider) and US$0.024 live.
2. **Benchmark the real code path.** Tool loops multiply calls (1.5 to 3.2 per turn here). A model that calls more tools costs more than its per-token price suggests.
3. **Test the instrument before trusting it.** In September a 6-case eval gave the strongest model the lowest score. The bug was in the eval: it checked figures against the tools' arguments instead of their results, so every price a tool returned looked invented. It was fixed before this benchmark.
4. **Read the errors.** Two models answered HTTP 400 because the agent switched reasoning off and they require it. Without reading the error they would simply have looked broken.
5. **Capacity is part of the result.** A model whose provider rate-limits you 80% of the time is not a candidate, however good it looks.
6. **Judges can cost more than the candidates.** Judging was US$5.7 of US$8.9, and one judge returned no usable verdict on 13 of the 30 turns. Next time: cheaper judges, or low reasoning effort for them.

## Run the public version

The production harness can't be published: it imports the private agent code and reads customer data. `benchmark.py` keeps the skeleton of the method so you can compare models on your own: the same conversations to every model, cost from OpenRouter's billing data, latency, tokens, prompt-cache hits, and automatic checks on every reply.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python3 benchmark.py --dry-run        # validate the data and print the plan: no key, no network
python3 benchmark.py --show-prompt    # the fictional store's system prompt

export OPENROUTER_API_KEY=...         # your own key
python3 benchmark.py --models anthropic/claude-haiku-5.5,openai/gpt-6-luna --max-cost 0.25
```

Each run writes `runs/<timestamp>.csv`, one row per reply (latency, prompt, cached and completion tokens, billed cost, provider, figures found, failed checks, and the reply text), and `runs/<timestamp>-summary.csv` with one row per model.

**The checks** (heuristics, so read the replies before trusting a ranking):

- **Unlisted figures.** Any money figure that isn't a catalog price or fee times 1–10, or a known total for that case, is flagged. It plays the role of the production verifier that rejects ungrounded prices.
- **Must / must-not patterns.** The right price is present; the injected 30% discount isn't applied; no bank account number; the canary code in the system prompt never leaks; no promise of an instant human handoff.
- **Length.** A two-word question gets a short answer.

**How it differs from the production evaluation:** no tools (the catalog is in the prompt), one call per turn, a prompt of about 1k tokens instead of 33–57k, invented data and no LLM judge. Absolute costs will be far lower than in production; use it to compare models against each other and to practice the method. The 20 cases × 6 default models make 120 short calls, which should cost cents; `--max-cost` (default US$1) stops the run regardless. Default model IDs are the OpenRouter IDs of October 2026; models get renamed and retired, so check [openrouter.ai/models](https://openrouter.ai/models).

`--reasoning`, `--repeat`, `--only`, `--temperature` and `--allow-data-collection` are documented in `python3 benchmark.py --help`. By default requests are routed only to providers that don't store or train on prompts.

## How this was built: with Claude Code

I work AI-first: Claude Code writes most of the code, and I direct it, review it and own the result.

- **My part:** the question (the cheapest model that holds correctness, tone and tool use on real traffic), the constraints (read-only production data, every side effect stubbed, a spend cap per model, blind judges from two vendors, the production model as baseline), reviewing the numbers, and the decision. The switch went to production on 2026-10-10.
- **Claude Code's part:** it wrote the replay harness, the judging and analysis scripts, ran them, and summarized the results. It also wrote this repository (the fictional store, the cases, `benchmark.py` and this README) from the evaluation's aggregate results, and I reviewed it.
- **Verification:** `benchmark.py` was tested offline (data validation, unit checks of the rule logic, and a mocked end-to-end run that covers retries, HTTP errors and the spend cap). No paid API calls were made to build this repository.

## Data and privacy

`results/` holds aggregates per model only: no customer messages, no system prompt, no store prices, no internal URLs. Everything in `data/` (store, catalog, prices, policies and conversations) is invented.

## License

[MIT](LICENSE) © 2026 Jhonatan González · [github.com/1914Jegx](https://github.com/1914Jegx)

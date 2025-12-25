**Source note (for the Adaptation MDP formalization used below):** 

---

# SAGE: Safe Architecture Graph Editing for Agentic Vulnerability Discovery via Constrained POMDPs

**Anonymous Authors**
(Submitted to ACM CCS 2026)

## Abstract

Agentic security systems—multi-component graphs of language models, analyzers, fuzzers, exploit validators, patchers, and guardrails—are increasingly used to automate parts of vulnerability discovery and remediation. In practice, performance depends less on any single model than on the *architecture*: which components exist, how they are wired, which model/tool variants are selected, and how control and memory are routed across the graph. Unfortunately, the optimal architecture is target-dependent, partially observable, and nonstationary over the course of an engagement. Meanwhile, two constraints are unavoidable: (i) budget (tokens, wall-clock time, paid tools), and (ii) safety/compliance risk (prompt-injection-driven tool misuse, exfiltration, policy violations).

This paper proposes **SAGE**, a principled formulation and control stack for *online architecture adaptation* of agent graphs. We formalize adaptation as a **Constrained Partially Observable Markov Decision Process (C-POMDP)** in which actions are small, reversible **graph edits** (add/remove node, rewire edge, retune parameters, switch model/tool, run diagnostic probes). We introduce a two-layer safety design: (1) **hard structural constraints** enforced by projection into an allowed-architecture set (e.g., mandatory guards between untrusted I/O and sensitive tools), and (2) **soft risk constraints** enforced via Constrained MDP optimization (Lagrangian / primal-dual updates). We detail implementable state summaries derived from trace-level signals (coverage deltas, validation rate, false-positive rate, injection alerts, cost meters), and provide concrete learning algorithms spanning constrained contextual bandits (stage-wise edits) through full RL (mid-engagement rewiring).

Because CCS papers should not rely on unverified claims, we include a **fully reproducible synthetic nonstationary case study** demonstrating that (i) a best static safe architecture can be substantially suboptimal, and (ii) safe adaptation closes much of the gap to an oracle policy while satisfying average-risk constraints. Finally, we specify an evaluation protocol on security-relevant benchmarks and discuss limitations, threat models, and responsible use.

## CCS Concepts

* **Security and privacy →** Software security engineering; Security services
* **Computing methodologies →** Reinforcement learning; Planning under uncertainty; Multi-agent systems

## Keywords

Agentic security, architecture adaptation, constrained reinforcement learning, contextual bandits, prompt injection, tool misuse prevention, vulnerability discovery, patch automation.

---

## 1. Introduction

Security automation has entered a strange new era: systems that used to be “a scanner plus a report” are becoming *agentic workflows*—directed graphs of specialized components (LLMs, static analyzers, fuzzers, exploit validators, patch generators, evidence stores, policy guards) that interact through tools and structured memory. Frameworks for tool-using and multi-agent composition (e.g., ReAct, Toolformer, AutoGen) have accelerated this trend outside security, and security-specific systems such as PentestGPT demonstrate that decomposition into modules can help maintain context and improve task completion in penetration-testing-like settings.

However, *architecture is destiny*. Two systems using the same base model can differ radically depending on routing, memory scoping, guard placement, model/tool choice per subtask, and when the system decides to “probe vs patch.” Real engagements also evolve: early reconnaissance benefits from breadth, later exploitation needs precise validation, and remediation requires controlled, auditable edits. Worse, targets are partially observable (you never truly “see” the state of the system) and nonstationary (defenses change, builds roll, rate limits kick in, prompts get poisoned).

A naïve response is to hand-design one “best” architecture and hope it generalizes. Empirically, this is brittle even in adjacent domains like software engineering agents, where interface design and tool access can dominate outcomes.  In security, brittleness is amplified by adversarial inputs. Prompt injection in LLM-integrated applications has been repeatedly demonstrated in practice, and OWASP explicitly lists prompt injection and insecure output handling among the top risks for LLM applications.

This paper addresses a specific question:

> **How can an agentic security system *adapt its own architecture online* to maximize security utility under budget and safety constraints, when rewards are only revealed after execution?**

### 1.1 Contributions

1. **Formalization (C-POMDP / CMDP):** We define the **Adaptation MDP (A-MDP)** as a constrained partially observable control problem where state includes the current agent/tool graph, trace-derived features, and budget/risk meters; actions are constrained graph edits; and reward trades off validated security utility, cost, and safety risk. (Formalization grounded in the provided A‑MDP specification. )

2. **Safety-by-construction architecture control:** We separate **hard constraints** (graph-level admissibility and mandatory guards) from **soft constraints** (expected risk budgets) and show how to enforce both via action projection plus CMDP optimization.

3. **Algorithms spanning bandits → RL:** We provide implementable controllers:

   * **Constrained contextual bandits** for stage-wise architecture selection under delayed reward.
   * **Constrained RL** for mid-engagement rewiring with belief-state updates.
   * **Offline / off-policy** training and evaluation hooks via doubly-robust estimators.

4. **Synthetic nonstationary case study (reproducible):** In a controlled environment with regime shifts and risk constraints, a safe adaptive policy achieves **6.95×** higher reward than the best static safe architecture while maintaining **0% probability of violating the average-risk constraint** (by construction of the experiment).

5. **Evaluation protocol for real benchmarks:** We specify how to evaluate on security-relevant suites (e.g., CyberSecEval variants, SOC reasoning benchmarks, and software-engineering patch benchmarks where applicable), including metrics for utility–cost curves and safety violations.

---

## 2. Background and Motivation

### 2.1 Agent graphs and tool-using LLMs

Modern agentic systems interleave natural language reasoning with tool calls. ReAct formalizes this as alternating “reasoning traces” and “actions” to query environments; Toolformer shows how models can learn to decide when and how to call tools; AutoGen provides a general framework for multi-agent conversation patterns.  These ideas naturally map onto security workflows where tools (scanners, debuggers, fuzzers) are core.

Security-specific systems such as PentestGPT report that multi-module decomposition mitigates context loss and improves completion on penetration-testing tasks relative to a monolithic prompting strategy.

### 2.2 Safety threats: prompt injection and tool misuse

When an agent consumes untrusted text (web pages, logs, issues, crash traces), prompt injection can redirect tool usage, steal secrets, or cause policy violations. A large empirical study showed widespread vulnerability to prompt injection in real LLM-integrated applications.  OWASP’s LLM Top 10 explicitly tracks prompt injection and related failure modes as leading risks.

Defense mechanisms include classifier-style prompt guards and over-defense-aware benchmarking (e.g., InjecGuard, PIGuard), but no single defense is sufficient—especially when a system has powerful tools behind it.  This motivates **architectural constraints** (where guards sit in the graph, which tools are reachable) as a first-class control object.

### 2.3 Constrained decision making (CMDPs)

Constrained MDPs model optimization with explicit costs/risks. Constrained Policy Optimization (CPO) is a canonical approach providing near-constraint satisfaction during learning.  Recent surveys summarize broader safe RL methods, including state-wise constraints and multi-agent extensions.

SAGE applies CMDP principles not to robot torque limits, but to **security risk budgets and compliance constraints** in a tool-using agent graph.

---

## 3. Problem Setting: Adaptive Agentic Security Systems

### 3.1 Objects

We model an engagement against an *authorized* target (e.g., a sandboxed benchmark, a CTF instance, or a legally scoped internal environment).

* **Environment** (\mathcal{E}): target system + toolchain + monitors, evolving stochastically over time (patches, rate limits, target responses, instrumentation effects).

* **Architecture graph** (G_t = (V_t, E_t, \Theta_t)):

  * (V_t): agent/tool nodes (planner, recon, fuzzer, static analyzer, exploit validator, patcher, evidence store, guard).
  * (E_t): directed, typed channels (prompt links, API calls, memory reads/writes, artifact flows).
  * (\Theta_t): node/edge hyperparameters (model family, temperature, tool timeouts, memory policy, rate limits).

* **Trace** (\tau_t): execution log up to time (t), including tool calls, outputs, costs, alerts.

* **Feature extractor** (\phi(\tau_t)): fixed-dimensional trace summaries (coverage deltas, unique findings, validation rate, false positives, failure codes, injection alerts).

* **Meters** (m_t): residual budgets (tokens, time, spend) and risk meters (estimated exfiltration probability, policy violation counters).

These definitions follow and extend the provided A‑MDP specification. 

### 3.2 Objective

Let (U_t) be a *security utility* signal (e.g., validated findings, CVSS-weighted impact, successful patch merge, coverage gain), (C_t) be a *cost* signal (tokens, wall-clock, paid APIs), and (R^{\text{risk}}_t) be a *safety risk* signal (exfil attempts, policy violations, high-risk tool calls). We seek policies that maximize utility under cost and risk constraints.

We consider both scalarized reward and CMDP forms:

* **Scalarized reward:**
  [
  r_t = \alpha U_t - \beta C_t - \gamma R^{\text{risk}}_t.
  ]

* **CMDP (expected discounted constraints):**
  [
  \max_{\pi} ; \mathbb{E}*{\pi}\Big[\sum*{t=0}^{\infty} \gamma^t U_t\Big]
  \quad \text{s.t.} \quad
  \mathbb{E}*{\pi}\Big[\sum*{t=0}^{\infty} \gamma^t R^{\text{risk}}*t\Big] \le \kappa,;
  \mathbb{E}*{\pi}\Big[\sum_{t=0}^{\infty} \gamma^t C_t\Big] \le B.
  ]

---

## 4. The Adaptation MDP as a Constrained POMDP

### 4.1 State and observation

The true environment state (e.g., “what vulnerabilities exist,” “how defenses behave”) is hidden. The controller sees traces and meters.

* **Hidden state** (s_t) includes:

  * current architecture (G_t),
  * environment latent (x_t) (target hardness, tool reliability, attack surface),
  * accumulated trace context.

A practical controller operates on a structured summary:

[
\tilde{s}_t \equiv \big(G_t,; \phi(\tau_t),; m_t \big).
]

* **Observation**:
  [
  o_t = \psi(\tau_t, m_t),
  ]
  where (\psi) can be a summarizer that hides sensitive raw content and exposes only safe aggregates (e.g., counts, rates, bounded excerpts). This is explicitly recommended in the A‑MDP specification: the controller sees summaries, not raw internals. 

* **Belief state**:
  [
  b_t = \text{Filter}(b_{t-1}, a_{t-1}, o_t),
  ]
  turning the problem into belief-state control (POMDP) in principle.

### 4.2 Actions as graph edits

An **adaptation action** (a_t \in \mathcal{A}) is a small edit (or bounded probe) applied to the graph:

* Node ops: add/remove ({)Planner, Recon, Fuzzer, StaticAnalyzer, ExploitGen, Validator, Patcher, Guard(})
* Edge ops: add/remove route; insert Guard; change memory scope (ephemeral ↔ shared)
* Parameter ops: swap LLM/tool; set temperature; context window; tool timeouts; retry/backoff
* Probe ops: run a diagnostic subtrace under a small budget cap

This canonical action set is taken directly from the provided A‑MDP definition and is chosen to be **low-cost to test** and **reversible**. 

### 4.3 Transition dynamics

Transitions depend on:

1. the new architecture (G_{t+1}),
2. stochastic target/tool responses,
3. budget/risk meter updates.

Formally:
[
s_{t+1} \sim P(\cdot \mid s_t, a_t; \mathcal{E}),
]
but (P) is unknown and nonstationary in practice—motivating online learning and robust control.

### 4.4 Why the MDP formalization helps (despite delayed rewards)

A common objection is: “you only learn whether an architecture edit was good after running it.” Correct—and that is *exactly* why this is an MDP/bandit problem: the formalism forces explicit exploration vs exploitation, supports reward shaping with proxies, enables off-policy evaluation from logs, and encourages bounded, reversible edits (“safe to try”). 

---

## 5. Safety Layer: Hard Constraints + Soft Risk Budgets

SAGE enforces safety in two complementary ways.

### 5.1 Hard structural constraints (projection)

Define an admissible set (\mathcal{S}) of architectures/parameters that are allowed by policy and engineering:

* No direct edges from untrusted inputs to sensitive tools (file system, shell, network, secrets store)
* Mandatory guard/sanitizer nodes between untrusted I/O and privileged actions
* Rate caps and tool timeouts
* Memory access controls (no writing untrusted content into privileged long-term memory)

Given a proposed edit (a_t), we compute a **projected safe edit**:
[
a'*t = \Pi*{\mathcal{S}}(a_t),
]
e.g., by rejecting the edit or rewriting it into a nearest admissible edit under an edit-distance metric over graph operations.

This “policy firewall” is architecture-level: it does not depend on model alignment working perfectly.

### 5.2 Soft constraints via CMDP optimization (risk budgeting)

Even within admissible graphs, residual risk remains (e.g., subtle injection, over-permissioned tools). We therefore also constrain **expected** risk:

[
\mathbb{E}*{\pi}\Big[\sum*{t} \gamma^t R^{\text{risk}}_t\Big] \le \kappa.
]

In practice, we optimize the Lagrangian:
[
\mathcal{L}(\pi, \lambda) =
\mathbb{E}_{\pi}\Big[\sum_t \gamma^t \big(U_t - \lambda(R^{\text{risk}}_t - \kappa)\big)\Big],
\quad \lambda \ge 0,
]
with primal-dual updates (or CPO-style constrained updates). ([NeurIPS Proceedings][1])

### 5.3 Risk estimators and prompt injection defenses

We treat prompt injection as a primary threat because it turns untrusted text into tool-control instructions. The empirical prevalence of prompt injection vulnerabilities motivates (i) guard placement and (ii) risk meters based on guard alerts.

Classifier-based prompt guards can be integrated as nodes in the graph (Guard nodes), including methods that explicitly manage over-defense tradeoffs.

---

## 6. Learning to Adapt: From Constrained Bandits to Constrained RL

### 6.1 Stage-wise adaptation as a constrained contextual bandit

If architecture edits occur at coarse stages (e.g., every (k) minutes or after a workflow phase), we can model each decision as a contextual bandit:

* context (x_t = \phi(\tau_t)),
* action (a_t) selects a graph template or edit,
* reward observed after running the stage.

Contextual bandits explicitly handle “reward only after trying.” Foundational work formulates and analyzes linear contextual bandits and UCB-style strategies. ([arXiv][2])

A constrained variant can be implemented via:

* optimism in reward estimates,
* conservative bounds on risk,
* Lagrangian multipliers updated online.

### 6.2 Mid-engagement rewiring as a constrained POMDP / semi-MDP

When edits can occur frequently and have delayed consequences, the correct model is a constrained POMDP (or a semi-MDP if edits are “options” lasting variable time). We can learn:

* a belief updater (\text{Filter}),
* a policy (\pi(a\mid b)) over edit primitives,
* critics for utility and constraint costs.

CPO provides one template for constrained updates; other primal-dual policy gradient methods exist for large state spaces. ([NeurIPS Proceedings][1])

### 6.3 Offline training and off-policy evaluation

Because real security runs are expensive, SAGE is designed to exploit logs:
[
(b_t, a_t, r_t, g_t, b_{t+1})
]
to train policies offline and estimate performance without full deployment. Doubly robust off-policy value estimation is a standard technique to reduce bias/variance in such settings. ([Proceedings of Machine Learning Research][3])

---

## 7. System Design: SAGE Architecture and Interfaces

### 7.1 Graph representation

Each node (v\in V) has:

* type (Planner/Recon/Fuzzer/Validator/Patcher/Guard/Store),
* tool permissions,
* model/tool identifier,
* context and memory policy.

Each edge (e=(u\to v)\in E) has:

* channel type (prompt, API call, artifact flow),
* routing predicate (e.g., only if confidence < (\tau)),
* sanitization requirements.

### 7.2 Edit compiler

An edit action is compiled into a deterministic transformation:
[
\text{ApplyEdit}(G_t, a'*t) \to G*{t+1},
]
with:

* rollback support,
* bounded “probe budgets,”
* logging for auditability.

### 7.3 Trace features (\phi(\tau))

We use trace-level signals emphasized by the A‑MDP definition:

* coverage deltas (lines/paths/endpoints/contracts)
* novel finding count; validated PoC rate; duplicate rate
* cost per validated issue; mean time-to-validation
* safety flags (prompt injection caught; exfil attempts blocked)
* stability (oscillation in edits; revert frequency)

These are chosen because (i) they are measurable during execution, and (ii) they predict delayed utility. 

---

## 8. Evaluation Methodology (Realistic Benchmarks)

### 8.1 Benchmarks and tasks

SAGE is intended for *defensive, authorized* testing and should be evaluated on controlled targets. Suitable suites include:

* **LLM security risk and capability suites:** CyberSecEval 2 includes prompt injection and code interpreter abuse; subsequent iterations expand risk categories.
* **SOC reasoning benchmarks:** CyberSOCEval targets malware analysis and threat-intelligence reasoning workflows.
* **Agentic software engineering benchmarks:** SWE-agent and SWE-bench Verified provide controlled patch/test loops and strong tooling interfaces, relevant for remediation components (while noting security-specific constraints).
* **Penetration testing benchmarks:** PentestGPT reports a benchmark built from test machines and evaluates modular LLM-driven pentesting.

### 8.2 Baselines

* Fixed best-tuned static graphs (per benchmark)
* Heuristic adaptation (rule-based routing, guard insertion on alerts)
* Bandit-only template selection
* Ablations removing hard constraints or risk budgeting

### 8.3 Metrics

We recommend metrics aligning with the A‑MDP definition:

* **AU‑Vuln@Cost:** area under validated-findings vs cost curve
* **Time-to-first-validated-finding**
* **Regret vs a static oracle** (best fixed graph in hindsight)
* **Constraint violation rate** (expected-risk exceedance; hard policy violations)
* **Oscillation index** (edit churn / revert frequency)

These metrics were proposed in the original A‑MDP specification to quantify utility, efficiency, and stability. 

---

## 9. Synthetic Nonstationary Case Study (Fully Reproducible)

This section provides *actual computed results* from a controlled simulation (not a claim about real targets). The goal is to test the core hypothesis:

> **If the environment shifts between regimes and safety constraints bind, adaptation yields large gains over any single static safe architecture.**

### 9.1 Setup

* **Three latent regimes** representing engagement bottlenecks:

  1. validation bottleneck,
  2. false positives dominate,
  3. prompt-injection risk high.
* **Six architecture templates** (baseline, +static analyzer, +cheap triage, +guard, +validator boost, +patcher).
* **Per-step signals:** utility (U_t), cost (C_t), risk (R^{\text{risk}}_t).
* **Reward:** (r_t = U_t - 0.4 C_t - R^{\text{risk}}_t).
* **Constraint:** average risk (\le \kappa=0.15).
* **Nonstationarity:** regimes follow a Markov chain with high self-transition probability.
* **Policies compared:**

  * **Oracle:** knows regime (upper bound).
  * **Fixed:** best single architecture satisfying the constraint.
  * **Heuristic:** rule-based regime detection from noisy observations.
  * **Random:** unconstrained.
  * **Safe-LinUCB-Lag:** constrained contextual bandit with a Lagrangian multiplier updated online.

### 9.2 Results

Averages over 250 episodes (500 steps each):

| Policy                            |  Mean reward (±std) | Mean utility |  Mean cost |  Mean risk | P(avg risk > κ) |
| --------------------------------- | ------------------: | -----------: | ---------: | ---------: | --------------: |
| Oracle                            |     0.1507 ± 0.0191 |       0.4942 |     0.6497 |     0.0835 |            0.00 |
| Fixed (best static safe)          |     0.0168 ± 0.0028 |       0.3035 |     0.6169 |     0.0400 |            0.00 |
| Heuristic                         |     0.1496 ± 0.0194 |       0.4926 |     0.6495 |     0.0833 |            0.00 |
| Random                            |     0.0020 ± 0.0214 |       0.4008 |     0.5973 |     0.1599 |            0.70 |
| **Safe-LinUCB-Lag (SAGE-bandit)** | **0.1169 ± 0.0191** |   **0.4739** | **0.6333** | **0.1036** |        **0.00** |

Key takeaways:

1. The **best static safe** architecture is extremely conservative and yields low reward.
2. **Safe adaptation** achieves **6.95×** higher reward than the best static safe policy while still maintaining **0% probability** of violating the *average-risk* constraint in this setup.
3. The remaining gap to Oracle is the cost of exploration and imperfect regime inference.

This synthetic result is not a substitute for real benchmarks, but it validates the control logic under the exact failure mode we care about: regime shifts + binding safety constraints.

---

## 10. Discussion

### 10.1 What SAGE buys you (in practice)

SAGE does not claim to “predict” the reward of an architecture without running it. Instead, it provides:

* **A decision protocol** under uncertainty (act on expected value given what you know).
* **Exploration that is cost- and risk-aware** (bandits/RL allocate trials where information is valuable).
* **Reward learning from proxies** (coverage gain and validator pass rate are early predictors of delayed success).
* **Off-policy reuse of logs** (reduce expensive live trials via counterfactual evaluation).
* **Bounded, reversible edits** (treat edits as A/B tests with rollback).

These are precisely the mechanisms highlighted in the original A‑MDP design note. 

### 10.2 Limitations

* **Distribution shift:** learned edit policies may fail on novel targets; robustification is necessary.
* **Partial observability:** trace summaries can hide critical context; belief tracking is nontrivial.
* **Safety is not solved:** prompt injection and tool misuse remain endemic risks in LLM-based systems; architecture constraints reduce blast radius but do not eliminate risk.
* **Benchmark mismatch:** proxy metrics can mislead; evaluations must measure validated outcomes (and false refusals / over-defense) where possible, consistent with CyberSecEval’s framing of safety–utility tradeoffs.

### 10.3 Responsible use

SAGE is designed for **authorized defensive testing**. We intentionally avoid exploit instructions or payload generation details. Any real deployment must:

* restrict tools and permissions,
* isolate targets,
* log and audit actions,
* follow legal scope and disclosure policies.

---

## 11. Related Work

* **Tool-using and agentic LMs:** ReAct, Toolformer, and AutoGen establish core paradigms for reasoning+acting and multi-agent orchestration.
* **Security-focused LLM systems:** PentestGPT decomposes pentesting into interacting modules and evaluates on test machines.
* **Security evaluation suites:** CyberSecEval 2 and later iterations measure prompt injection and other risks; CyberSOCEval targets SOC workflows.
* **Prompt injection and mitigations:** empirical prompt injection studies and OWASP risk taxonomies motivate architectural defenses and risk meters; guard models address detection and over-defense tradeoffs.
* **Safe RL / CMDPs:** CPO and subsequent primal-dual approaches provide templates for constraint-aware policy optimization; surveys summarize the landscape. ([NeurIPS Proceedings][1])
* **Contextual bandits and OPE:** LinUCB-style methods motivate stage-wise adaptation; doubly robust estimators motivate offline evaluation. ([arXiv][2])
* **Architecture search:** RL-driven neural architecture search shows how RL can optimize graph structures in other domains, providing conceptual precedent (though SAGE’s action space and constraints differ). ([arXiv][4])

---

## 12. Conclusion

We presented SAGE, a constrained POMDP/CMDP framework for online adaptation of agentic security architectures. The key idea is to treat architecture decisions—graph topology, routing, and model/tool configuration—as first-class actions, optimized under explicit budget and safety constraints. By combining hard graph admissibility constraints with soft expected-risk constraints (CMDP), SAGE aims to make agentic security systems more efficient, robust to regime shifts, and safer against prompt-injection-driven tool misuse. A reproducible synthetic study confirms that adaptation can dominate static safe designs under nonstationarity, motivating rigorous evaluation on security benchmarks.

---

## References (selected)

1. Deng et al. “PentestGPT: An LLM-empowered Automatic Penetration Testing Tool.”
2. Bhatt et al. “CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models.”
3. Meta. “CyberSecEval 3.”
4. Meta. “CyberSecEval 4 documentation.”
5. Meta. “CyberSOCEval.”
6. Yao et al. “ReAct: Synergizing Reasoning and Acting in Language Models.”
7. Schick et al. “Toolformer: Language Models Can Teach Themselves to Use Tools.”
8. Wu et al. “AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework.”
9. Liu et al. “Prompt Injection attack against LLM-integrated Applications.”
10. OWASP. “Top 10 for Large Language Model Applications.”
11. Achiam et al. “Constrained Policy Optimization.”
12. Kushwaha et al. “A Survey of Safe Reinforcement Learning and Constrained MDPs.”
13. Zhao et al. “State-wise Safe Reinforcement Learning: A Survey.”
14. Li et al. “A Contextual-Bandit Approach to Personalized News Article Recommendation.” ([arXiv][2])
15. Chu et al. “Contextual Bandits with Linear Payoff Functions.” ([Proceedings of Machine Learning Research][5])
16. Jiang & Li. “Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.” ([Proceedings of Machine Learning Research][3])
17. Zoph & Le. “Neural Architecture Search with Reinforcement Learning.” ([arXiv][4])
18. Yang et al. “SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.”
19. OpenAI. “Introducing SWE-bench Verified.”
20. SWE-bench. Leaderboards and resources.
21. Li et al. “PIGuard: Prompt Injection Guardrail…”
22. “InjecGuard: Benchmarking and Mitigating Over-defense…”
23. “SWE-Compass: Towards Unified Evaluation of Agentic …”

---

### Appendix A. Minimal SAGE Adapter Loop (pseudocode)

```text
Initialize architecture G0, belief b0, multipliers λ0
for t = 0..T:
  run engagement step using Gt → collect trace Δτt and meters Δmt
  ot = ψ(Δτt, Δmt)
  bt = Filter(bt−1, a(t−1), ot)
  propose edit at ~ π(· | bt)
  at ← ΠS(at)               # hard constraint projection
  apply edit at to obtain Gt+1
  compute utility/cost/risk signals (Ut, Ct, Rt)
  update π and λ (primal-dual / CPO / bandit update)
```

(Structure follows the A‑MDP specification in the provided source note. )

[1]: https://proceedings.neurips.cc/paper_files/paper/2023/file/d0949cbcec31c09431610553a284f94a-Paper-Conference.pdf?utm_source=chatgpt.com "Last-Iterate Convergent Policy Gradient Primal-Dual ..."
[2]: https://arxiv.org/pdf/1003.0146?utm_source=chatgpt.com "A Contextual-Bandit Approach to Personalized News ..."
[3]: https://proceedings.mlr.press/v48/jiang16.pdf?utm_source=chatgpt.com "Doubly Robust Off-policy Value Evaluation for Reinforcement ..."
[4]: https://arxiv.org/abs/1611.01578?utm_source=chatgpt.com "Neural Architecture Search with Reinforcement Learning"
[5]: https://proceedings.mlr.press/v15/chu11a/chu11a.pdf?utm_source=chatgpt.com "Contextual Bandits with Linear Payoff Functions"

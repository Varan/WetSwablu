# WetSwablu

A reinforcement-learning agent for **[Gen 9] Random Doubles Battle** on [Pokémon Showdown][showdown], combining a self-play policy/value transformer, dynamic action-legality masking, imperfect-information search, and a Rust battle simulator with Python bindings.

**Historical ladder peak: #1 in Gen 9 Random Doubles Battle** (`gen9randomdoublesbattle`).

**Status:** work in progress. This public repository contains documentation; the implementation and trained weights remain private. The technical overview describes the system at the component level.

## Contents

- [Why Pokémon is a useful decision-making problem](#why-pokémon-is-a-useful-decision-making-problem)
- [Technical overview](#technical-overview)
- [Results and scope](#results-and-scope)
- [What has been happening in the Pokémon community](#what-has-been-happening-in-the-pokémon-community)
- [Why the implementation and weights are private](#why-the-implementation-and-weights-are-private)
- [How this relates to Showdown's bot policy](#how-this-relates-to-showdowns-bot-policy)
- [References](#references)

## Why Pokémon is a useful decision-making problem

Pokémon Showdown is an online battle simulator. In Random Doubles, players receive generated teams and usually control two active Pokémon at once. The format combines several challenges:

- **Simultaneous decisions.** Both players commit actions before seeing the opponent's choices for that turn.
- **Hidden information.** Unrevealed Pokémon, moves, items, and abilities create uncertainty about the opposing team.
- **Stochastic transitions.** Accuracy checks, damage rolls, critical hits, and secondary effects can produce different outcomes from the same choices.
- **Coupled actions.** A move and its target must be considered alongside the partner Pokémon's action. Two individually reasonable choices can make a poor joint decision.
- **Changing legal actions.** Fainting, forced switches, available targets, and move restrictions change which choices are valid.

The [Random Battles format explanation][random-formats] and [Showdown simulator protocol][sim-protocol] provide background on generated teams and battle choices.

For example, attacking an opponent with both active Pokémon may secure a knockout if it stays in, but waste an action if it protects or switches. The agent needs to evaluate coordinated choices under uncertainty, rather than simply select the move with the largest immediate damage.

## Technical overview

WetSwablu combines learned action selection with search and parallel simulation. The diagram below shows the conceptual decision path:

```text
Battle information available to the agent
                  |
                  v
        Policy / value transformer
                  |
                  v
       Legal candidate joint actions
                  |
                  v
   Search over possible continuations
   using simulation and learned evaluation
                  |
                  v
            Selected action
```

The simulator also supplies environments for self-play training. Network layout, observation encoding, search budget, and scheduling details may change as development continues.

### Self-play policy and value learning

The transformer is trained through self-play reinforcement learning. Its two central roles are:

- **Policy:** assess candidate actions from the agent's available battle information.
- **Value:** estimate how favorable a position is, providing an evaluation signal without requiring every considered continuation to reach the end of a battle.

Self-play supplies experience against the agent itself. Learning and search solve different parts of the problem: learning produces reusable action preferences and position evaluations, while search spends additional computation on the decision currently being made.

### Dynamic action-legality masking

The policy must distinguish a strategically poor action from an action the game does not permit. Masking excludes invalid choices from action selection as the battle changes.

Conceptually, a masked policy has the form:

```text
masked_probability(action)
    ∝ policy_probability(action) × is_legal(action)
```

The probabilities are normalized over legal choices. This illustrates the role of masking; it does not specify the network's action factorization or the exact implementation.

In doubles, legality also involves the combination of both slots' choices. For example, the two active Pokémon cannot both switch into the same reserve Pokémon. The [simulator protocol][sim-protocol] documents battle requests and choices; [Huang and Ontañón's paper][masking-paper] provides background on invalid-action masking in reinforcement learning.

### Search under uncertainty

WetSwablu uses imperfect-information search to consider possible continuations beyond the policy's immediate recommendation. The relevant uncertainties are different:

1. **Opponent decisions:** what the other player chooses this turn.
2. **Hidden state:** information about the opposing team that has not been revealed.
3. **Chance:** which random outcomes occur after both players act.

A simulator can advance a fully specified hypothetical battle, but that does not make its hidden information available to the real agent. Keeping the simulator's complete state separate from the information available to a decision-maker is an important design constraint.

Likewise, evaluating a turn as though the opponent commits first and reveals its action changes the simultaneous-decision problem. The Jaxcalibur, mikumiku37, and Nessie write-ups offer different approaches to these issues; they are useful context, not interchangeable search specifications. See the linked project descriptions below.

### Rust simulation and the Python boundary

The battle simulator is implemented in Rust and exposed to Python through [PyO3][pyo3]. It supports parallel game environments for reinforcement learning, reducing reliance on a Python-to-Node.js process boundary for repeated battle simulation.

The systems motivation is straightforward: self-play repeatedly needs state transitions. Communication, text parsing, and process management can consume time that could otherwise produce training experience. Showdown's [simulator API documentation][sim-api] describes its text-based interface and access from other languages through standard input/output.

The Rust/Python split keeps the simulation component native while making it accessible from the Python training side.

A useful performance comparison needs a defined workload, hardware, environment count, and a clear statement of whether inference is included. No numerical throughput or end-to-end speedup benchmark is published here.

## Results and scope

WetSwablu reached **#1 on the Gen 9 Random Doubles Battle ladder**. That is a historical project result, scoped to one format.

This repository does not currently include a public evaluation archive or a reproducible benchmark report. It does not claim sustained ladder dominance, superiority to the projects referenced below, or strength in formats that have not been evaluated.

The implementation remains under development. Simulator fidelity, uncertainty handling, and evaluation methodology remain important concerns alongside playing strength.

## What has been happening in the Pokémon community

*Context checked November 4, 2026. The results below are the respective authors' reports, not a shared benchmark or an evaluation of WetSwablu.*

Strong Pokémon bots predate this project. [Foul Play][foul-play] is a public Showdown bot, and its [poke-engine dependency][poke-engine] is a public Rust battle engine. The recent debate is about the competitive strength and accessibility of newer systems, rather than the first appearance of battle automation.

### Jaxcalibur brought the release question into focus

[Jaxcalibur's technical write-up][jaxcalibur-tech] describes a transformer trained through self-play, a JAX battle engine, and search adapted to hidden information and simultaneous decisions. Its author reported periods at #1 in Gen 9 Random Battles during August 2026, with a peak Elo of 2557.

The [creator's announcement][jaxcalibur-post] explicitly asked whether releasing a strong agent would benefit the community or facilitate cheating. In a **September 12, 2026 update**, the author decided not to release the code or weights, citing competitive integrity.

### Doubles projects showed that different approaches were viable

- **[mikumiku37][mikumiku]:** reported reaching #1 on the VGC Reg M-C ladder on October 1, 2026. The author described self-play training and a one-turn search that evaluates joint-action matchups across sampled worlds, then solves for a mixed strategy. The announcement withheld the engine and weights because of their potential competitive impact.
- **[Nessie][nessie]:** described a neural evaluator, search-guided self-play, matrix-game solving, and an analysis interface. Its October 3 announcement included a September 28 ladder peak. Its near-public-information training setup is an important difference from random-team formats.

These projects span different formats, information assumptions, training procedures, and compute budgets. Their results should not be treated as a cross-format leaderboard.

### The disagreement is about use as well as capability

The linked discussions contain both enthusiasm for better practice and analysis tools and objections to their effect on human competition. Nessie's thread is a particularly direct example: some participants welcomed learning resources, while others opposed live assistance and questioned whether moderation could keep pace. Participants also disagreed about how reliably assistance could be detected. There is no single community position represented by those replies.

For this project, the useful distinction is between **analyzing a completed game or practicing against a disclosed bot** and **consulting an agent during a competition intended to measure human decisions**.

## Why the implementation and weights are private

I want to document the engineering without distributing a ready-to-run competitive agent. The decision is based on how an agent could be used, not on a claim that WetSwablu is uniquely powerful or unbeatable.

### A bot does not need to control an account to provide assistance

A player could consult an agent and enter its recommendations manually. Removing automatic ladder connections would therefore leave a route to undisclosed assistance. A warning against cheating would state an expectation, but would not enforce it.

Jaxcalibur's release discussion identifies concerns about laddering, tournament assistance, and qualifying for suspect tests—the ladder-based qualification used for community votes on competitive rules. Those concerns help explain why publishing a complete agent is a different decision from publishing an architectural description.

### Legitimate uses deserve consideration too

Open source would make the work easier to inspect, reproduce, correct, and extend. Strong agents could also support practice, post-game analysis, and exploration of unfamiliar positions. The educational goals in the Nessie and mikumiku37 discussions are worth taking seriously.

There are costs to withholding the implementation: readers cannot independently reproduce this project's result from this repository, and other developers cannot directly build on it. I am accepting those limits for now. This repository is a public project overview, not a reproducible research release.

### Why both code and weights are withheld

Releasing code without a checkpoint would not provide the trained agent immediately. It could still lower the effort needed to train or adapt another competitive agent, particularly if it included the simulator, training pipeline, and search implementation.

For now, I am keeping the implementation and trained weights private together. This is a precaution about what this project distributes, not a claim that withholding one repository will prevent others from developing comparable systems. Public projects already exist, and the announcements above show that several approaches are viable.

The material here provides context for understanding the work without serving as a deployment package. There is no scheduled code or model release. Any future release would require a separate decision about the component being shared and its likely uses.

## How this relates to Showdown's bot policy

Smogon and Showdown have taken concrete steps to preserve human competition while allowing disclosed bot participation. Their [official policy thread][bot-policy] records:

| Date | Announcement |
| --- | --- |
| October 9, 2026 | Mandatory bot registration; support for optional bot play; protection of human competition; tournament AI assistance identified as ghosting. |
| October 12, 2026 | Registration opened, with a matchmaking control planned. |
| October 20, 2026 | An opt-in toggle for playing bots was added; battles against bots would not affect human ratings. |

## References

### Project and community sources

- [Jaxcalibur — technical write-up][jaxcalibur-tech]
- [Jaxcalibur — announcement, discussion, and September 12 release decision][jaxcalibur-post]
- [mikumiku37 — creator announcement and technical discussion][mikumiku]
- [Nessie — creator announcement and community discussion][nessie]
- [Foul Play — public bot repository][foul-play]
- [poke-engine — public Rust engine repository][poke-engine]
- [Smogon / Showdown — initial bot policy and dated updates][bot-policy]

### Relevant Background Information

- [Pokémon Showdown — Random Battles format explanation][random-formats]
- [Pokémon Showdown — simulator protocol][sim-protocol] and [simulator API][sim-api]
- [PyO3 — Rust bindings for Python][pyo3]
- [Huang and Ontañón — A Closer Look at Invalid Action Masking in Policy Gradient Algorithms][masking-paper]


[showdown]: https://pokemonshowdown.com/
[random-formats]: https://www.smogon.com/forums/threads/questions-about-how-random-battles-formats-work-read-here.3712694/
[sim-protocol]: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md
[sim-api]: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md
[pyo3]: https://pyo3.rs/main/index.html
[masking-paper]: https://arxiv.org/abs/2006.14171
[jaxcalibur-tech]: https://jaxcalibur.github.io/
[jaxcalibur-post]: https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/
[mikumiku]: https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/
[nessie]: https://www.smogon.com/forums/threads/nessie123-an-ots-vgc-bot-that-topped-the-reg-m-c-bo3-ladder.3789213/
[foul-play]: https://github.com/pmariglia/foul-play
[poke-engine]: https://github.com/pmariglia/poke-engine
[bot-policy]: https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/

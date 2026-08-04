This is the checklist of feedback that Tomas would provide as a human:

### Check Constraints
Are the constraints mentioned in the plan really necessary? Would relaxing the constraint improve the chances of a successful implementation? E.g.: A constraint in the plan says to "freeze dependencies" early, but on modal we can easily update dependencies, e.g. when upgrading CUDA to a new API would simplify or enable the target implementation.

## Artifacts
Always mention what the testing / validation for a step of the plan is. Mention where a human can find the artifacts that validate the step was implemented correctly. The plan should state what a human verifier should be checking: What was expected outcome? And the actual outcome? What could have failed? The surface for the reviewer should be clear to the agent implementing the plan. Don't commit generated artifacts in git.

## Break Work Down
Is this step perhaps too ambitious? We are tackling difficult problems. We would rather make slow progress than moving fast but failing. Can this step be broken down into smaller steps? Incremental progress keeps the agent focused on achieving a single thing, and means that if an attempt fails, less work is lost.

Continuously reassess whether the plan has the right task granularity as implementation work reveals new information. Split steps that are too broad to execute or validate reliably, and combine steps whose separation adds overhead without isolating a meaningful risk or decision. Update the remaining plan when that assessment changes instead of treating its initial structure as fixed.

## Document Next Steps
We work step by step. After each step is done we assume that a new agent with no context whatsoever could pick up the next step and work on it. This means in practice that the previous agent needs to document where it left off, what it believes are the most sensible next steps, and where to continue the work without pre-solving the next step.

## Prevent Sprawl
Working incrementally in steps means that a lot of intermediate code, runners, and artifacts will be created. This is good, but it can also lead to sprawl. The plan should mention how to keep the intermediate code and artifacts organized, and how to clean up after each step. This will help prevent confusion and make it easier to track progress.

## Remove Infrastructure Friction
Continuously monitor the full development loop, including environment and image creation, dependency setup, compilation, profiling, queueing, logging, and repeated validation work.
Treat avoidable latency, cache invalidation, duplicate work, unreliable orchestration, and poor observability as implementation problems to diagnose and fix when they appear, rather than passive waiting costs.
Record enough timing and cache evidence to distinguish necessary one-time setup from recurring friction, and keep the workflow fast for subsequent iterations.

### Speed up Experimentation Speed
If the experiments on Modal are running slowly, think about how they could be accelerated: Perhaps only a subset of the experiments yields most of the information. Perhaps the container image is failing the cache layers correctly, perhaps the pytorch / triton / vllm cache is not being hit, triggering unnecessary recomputation. Could some experiments be run in parallel rather than sequentially? This introduces a tradeoff, since an experiment might make the others unnecessary, but it speeds up experiemntation in wall-time. Regularly check what takes most time during experimentation, and introduce optimizations to speed them up and abstractions to simplify them.

## Explain Results Top-Down
After each step, explain its results in a top-down narrative before diving into evidence: start with the bigger picture (what question the step answered and why it mattered), then what was done, what happened, and why it matters for the next step. Use plain language; introduce jargon only after the plain statement. The detailed tables, packets, and coverage audits sit below that narrative as supporting evidence, not as a substitute for it. A reader should be able to understand the outcome and its implication for the project from the summary alone.

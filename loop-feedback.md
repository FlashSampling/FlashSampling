This is the checklist of feedback that Tomas would provide as a human:

### Check Constraints
Are the constraints mentioned in the plan really necessary? Would relaxing the constraint improve the chances of a successful implementation? E.g.: A constraint in the plan says to "freeze dependencies" early, but on modal we can easily update dependencies, e.g. when upgrading CUDA to a new API would simplify or enable the target implementation.

## Artifacts
Always mention what the testing / validation for a step of the plan is. Mention where a human can find the artifacts that validate the step was implemented correctly. The plan should state what a human verifier should be checking: What was expected outcome? And the actual outcome? What could have failed? The surface for the reviewer should be clear to the agent implementing the plan. Don't commit generated artifacts in git.

## Break Work Down
Is this step perhaps too ambitious? We are tackling difficult problems. We would rather make slow progress than moving fast but failing. Can this step be broken down into smaller steps? Incremental progress keeps the agent focused on achieving a single thing, and means that if an attempt fails, less work is lost.

Continuously reassess whether the plan has the right task granularity as implementation work reveals new information. Split steps that are too broad to execute or validate reliably, and combine steps whose separation adds overhead without isolating a meaningful risk or decision. Update the remaining plan when that assessment changes instead of treating its initial structure as fixed.

## Prevent Sprawl
Working incrementally in steps means that a lot of intermediate code, runners, and artifacts will be created. This is good, but it can also lead to sprawl. The plan should mention how to keep the intermediate code and artifacts organized, and how to clean up after each step. This will help prevent confusion and make it easier to track progress.

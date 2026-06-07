## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

### Codebase Specific Guidelines

- Never delete code or documentation that was used to produce published results. Instead, move those files to an `archives/` directory.

* **Keep documentation minimally sufficient but accurate**

  * Update `README.md` when you:

    * Add/remove scripts
    * Change file structure
    * Modify how experiments are run
  * Focus on *reproducibility*, not completeness. Someone (including future you) should be able to rerun key experiments from the README.
  * While renaming files or functions, update references in documentation and code comments to avoid confusion.

* **Stabilize experiment interfaces**

  * If you change how a script is executed (arguments, config format, entry points), update the usage instructions in `README.md`.
  * Prefer config files (YAML/JSON) over hardcoded parameters for experiments.

* **Document reusable components only**

  * Add docstrings to functions/classes/modules that are:

    * Reused across files, or
    * Non-trivial to understand
  * Docstrings should include:

    * What the component does
    * Inputs (types + meaning)
    * Outputs (types + meaning)
  * Skip over-documenting throwaway or one-off experiment code.

* **Inline comments**

  * Write one brief, single-line comment above each code stub (a function, or a logical block within a function) describing what it does — not how.
  * Keep them short and skimmable; the comment should let a reader follow the flow without reading the code line by line.
  * Don't comment every line, and don't restate obvious code.

* **Shell scripts and sbatch jobs**

  * All sbatch scripts live under `bash_scripts/<category>/` (e.g., `bash_scripts/infer/`, `bash_scripts/benchmark/`).
  * After creating any new `.sh` script, run `chmod u+x <script>` to make it owner-executable.
  * Pass runtime arguments via environment variables (e.g., `EXP_JSON_PATH=... sbatch script.sh`) rather than hardcoding paths inside the script.
  * Submit with: `sbatch bash_scripts/<category>/<script>.sh`
  * Slurm logs are written to `./slurm_logs/` (relative to `PROJECT_ROOT`). Make sure the directory exists before submitting: `mkdir -p slurm_logs`.

* Guidelines to doing commits and pushes to the repository:
  Whenever I ask you to commit, by saying "commit", you should follow the following guidelines:

  * **Never stage or commit files under `archives/`**. These are archived for reference only and should not appear in new commits.

  * Perform incremental commits that reflect logical units of work, such as:

    * Adding a new feature or experiment
    * Refactoring code for clarity or performance
    * Fixing a specific bug
    * Updating documentation for a specific change
    * Each commit should have a clear and concise message that summarizes the change.
    
  * **Keep commits structured for research iteration**

    * Write concise and simple commit messages (e.g., `add voxel pruning baseline`, `fix pose evaluation bug`).
    * Group related changes into a single commit.
    * Separate:

      * Experiment changes
      * Refactors
      * Bug fixes
    * You do not need perfect commit hygiene, but avoid dumping unrelated changes together.
    * Only commit config files (YAML/JSON) and shell scripts (.sh) if the structure changed — new/removed keys or parameters, structural changes, or new files. Do not commit if only values were tweaked (e.g., a run name, step count, or path).

  * **Preserve experiment traceability**

    * Avoid silently changing core logic used in prior results.
    * If behavior changes, note it in commit messages or a short changelog section in `README.md`.
    * When possible, keep configs or scripts that reproduce key results.
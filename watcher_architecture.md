# Watcher Architecture Design

**Objective:** Implement a "Watcher" system where a secondary reasoning model supervises the primary model during the benchmark. If the primary model deviates (Deviation > 0), the Watcher intervenes to guide it back to the correct path.

## 1. Core Logic

The core logic will be injected into `run_agent_loop` in `llm_bench_longchain.py`.

### Trigger Condition

We will use a composite condition to avoid false positives:

- `status` is one of `{"WRONG", "OFF"}`
- `drift_score > 0`

### The Supervisor Loop

When deviation occurs:

1.  **Pause** the main execution.
2.  **Snapshot** the current state using a _minimal_ context (last action, result, expected path, short error note). Avoid full history unless absolutely necessary (e.g. repeated failures).
3.  **Construct Watcher Prompt**:
    - Context: "You are a Supervisor for an automated agent."
    - Last Action: "Agent called X with args Y."
    - Result: "Result was Z."
    - Error: "System deviation: Expected A, got B."
    - Goal: "Provide a clear, instructional text output to guide the agent to the correct next step."
4.  **Call Watcher Model**: Send this prompt to the `watcher_model` API.
5.  **Inject Feedback**: Append the Watcher's response to the Primary Agent's message history as a USER message `[SUPERVISOR]: ...`.
6.  **Verify/Retry Cap**:
    - The Primary Agent gets a chance to correct.
    - We enforce a **max intervention retry cap per step** (e.g., 3 attempts). If the agent is still failing, we release the pause to strictly avoid infinite loops/stalls.
7.  **Fail Closed**: If the Watcher API call fails, log the error and resume normal execution immediately (do not block the benchmark).

## 2. API & Configuration Changes

### New Arguments

- `--watcher-model`: string (Name/ID of the model in LM Studio to use as the watcher).
- `--watcher-base-url`: string (Optional, defaults to same as base-url).
- `--enable-watcher`: boolean flag.

### LM Studio Setup

The user must have two models loaded. LM Studio's API usually routes based on the `model` field in the JSON body. We will assume the endpoint is the same but the `model` ID differs.

## 3. Implementation Plan (`llm_bench_longchain.py`)

1.  **ArgParse**: Add new arguments to `main`.
2.  **`run_agent_loop` Modification**:
    - Inject `watcher_model` into the function signature.
    - Inside the `while` loop, after `execute_tool` and status determination:
      - Check `exec_res['wrong_read_streak']` or `drift_score`.
      - If `deviation`:
        - Call `_run_watcher_intervention(...)`
    - **`_run_watcher_intervention`**:
      - Recursive or inner `while` loop.
      - Calls Watcher API.
      - Updates `messages`.
      - **CRITICAL**: We need to allow the Primary Agent to _generate a new response_ based on the feedback _without_ advancing the simulation logic (steps) incorrectly if we want to "fix" the current turn, OR we treat it as the next step.
      - _Refined Logic_: The user said "loop only ceases when agent 1 proceeds correctly". This implies the "Watcher" step is a meta-step.
      - We will count these correction turns separately or as standard steps? Ideally separately to track "Corrective Overhead".

## 4. Communication Protocol (Internal)

- **Primary Agent**: "I want to read `gamma/trace_350.log`."
- **System**: "Result: ... Content: ... Path: WRONG (Expected 379)" -> **PAUSE**
- **Watcher**: (Receives: "Agent read 350, expected 379. Content of 350 says 'Archived trace'.") -> "Output: You read an archived trace (350). The logs indicate you should look for the sequence. Try finding trace_379.log instead."
- **Primary Agent**: (Receives `[SUPERVISOR]: You read an archived trace...`) -> "Ah, okay. Tool: read_file 'gamma/trace_379.log'"
- **System**: "Path: OK" -> **RESUME**

## 5. Proposed Code Structure

```python
def ask_watcher(watcher_model, history, error_context, base_url):
    # ... builds prompt ...
    # ... requests.post ...
    return correction_text

# Inside run_agent_loop
if enable_watcher and exec_res['wrong_read_streak'] > 0:
    print(f"    -> [WATCHER] Intervention triggered...")

    while exec_res['wrong_read_streak'] > 0 and steps < MAX_STEPS:
        # 1. Get correction
        correction = ask_watcher(watcher_model, messages, last_error, base_url)

        # 2. Inject correction
        messages.append({"role": "user", "content": f"[SUPERVISOR]: {correction}"})

        # 3. Agent retry
        completion = call_model(primary_model, messages)
        # ... parse tool ...
        # ... execute tool ...
        # ... check status ...

        if status == "OK":
            break # Loop ceases
```

## Questions for Codex

1.  Should we strictly "undo" the failed step in the VFS state? (e.g. if they wrote a bad file, do we delete it? Or just let them fix it forward?) _Recommendation: Fix forward. Real world agents make mistakes and fix them._
2.  Should we count "Watcher Steps" against the `MAX_STEPS` limit? _Recommendation: Yes, otherwise infinite loops are possible._

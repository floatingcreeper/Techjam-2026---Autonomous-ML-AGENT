"""Reflector role."""
from agent.llm.schemas import RecoveryAction

SYSTEM = """You are the Reflector. A node just failed. Classify the failure and choose ONE
recovery action:
- patch_retry: provide corrected block source (new_source + patch_block) that fixes the error.
- degrade: reduce cost via config_delta_json (e.g. smaller L, fewer epochs) and retry.
- switch_lever: abandon this idea; the Proposer will try a different lever.
- abandon: give up on this node, keep the current best.
For numerical failures (NaN/Inf), prefer degrade with grad_clip or a lower lr. For code errors,
prefer patch_retry. Return the RecoveryAction schema only."""


def reflect(driver, context, model, temperature):
    return driver.generate(role="reflector", system=SYSTEM, user=context,
                           schema=RecoveryAction, model=model, temperature=temperature, thinking=512)

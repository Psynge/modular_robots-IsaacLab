# HSS-SRMR Isaac Lab Port — Progress Report

**Project:** RL for 3-module Homogeneous Solid-State Self-Reconfiguring Modular Robots (HSS-SRMR), ported from MuJoCo to NVIDIA Isaac Lab. Target venue: CoRL 2026.
**Repo:** github.com/Psynge/modular_robots-IsaacLab (port) · github.com/Psynge/modular-robots-mujoco (original)
**Date:** July 18, 2026

## Why the port

The MuJoCo implementation runs one environment per process with a custom PPO loop; training plateaued and iteration was slow. Isaac Lab runs thousands of vectorized environments on the GPU with a maintained PPO stack (rsl_rl), which both accelerates training by orders of magnitude and fixes a structural weakness of the old loop (networks were rebuilt every run-set, so learning never compounded). A second motivation is scientific: the project's central physics question — whether a module can roll between magnet detents — is friction- and contact-dominated, exactly where simulators disagree. Reproducing the system under PhysX gives a cross-engine validation story rather than results tied to one engine's contact model.

## System architecture

The physics model is deliberately simulator-agnostic. All hand-verified constants live in two files that both engines' versions agree on: module geometry (hex module, 40mm across flats, six face-magnet sites, the verified y-axis spawn line) and the magnet law (inverse-square site-to-site force, 30mm cutoff, 4mm latch threshold with force capped at the value it would have at 4mm, 1mm singularity floor, 0.1g gravity). The electromagnet model itself was ported as a pure-math batched torch class: it takes world site positions and centers of mass in, returns force/torque wrenches out, with a leading `num_envs` dimension on everything so the same code that runs one probe environment will run 4096 training environments. The simulator side applies those wrenches externally each step; nothing about the model knows PhysX exists, which is what made it testable against the analytic formula in isolation.

Work proceeds by an agreed verification ladder, one probe at a time, nothing trusted until its output is inspected: (1) asset conversion and spawn stability, (2) force-model exactness, (3) seed-hold under gravity, (4) rotation, (5) full RL environment and training. Steps 1–3 are verified; step 4 is where the science currently lives.

## Progress and results

**Step 1 — geometry (verified).** The STL converts to USD and three modules spawn in the verified face-normal line, seat flush with site pairs at ~2mm, and sit motionless for 2000 steps with sub-millimeter drift. PhysX accepts the MuJoCo-phase geometry fix exactly.

**Step 2 — force model (verified, exact).** A gravity-off probe teleported two modules across separations from 29mm down to 3mm and compared three numbers per case: the analytic formula, the model's output, and the force PhysX actually realized (mass × velocity change over one step). All three agreed to every printed digit across attraction, repulsion, and both contact-cap branches. The wrench pipeline is exact.

**Step 3 — seed hold (verified, one finding).** With the four-magnet seed hold energized, the line stays latched for 2000 steps with steady capped hold forces — the failure mode that plagued MuJoCo training (assembly collapsing at step 0) does not occur when the hold is maintained, strengthening the suspicion that the old training loop's per-step CLEAR of the hold was the culprit. One finding: the outer modules settle at ~2.4° pitch. This is the expected equilibrium under hold torque (magnet sites sit ~2mm above the center of mass), but the magnitude is suggestively close to the 2° face chamfer, which has never been verified to function as designed.

**Step 4 — rotation (in progress; first result in hand).** The oscillation probe reproduces the MuJoCo recipe: hold one seam fixed, alternate the other module's latch between two adjacent detents at full strength. Result over 12 phases: the roll never completes. De-energizing the current latch and energizing the target breaks the latch and starts the rotation, but the module comes to a genuine static equilibrium 1.4–8.2° into the 60° detent — settled, not jittering. The torque is real and motion initiates; something arrests it. Consistent with the project's own ground rule, this shows this actuation recipe fails, not that rotation is impossible.

## Challenges encountered and solutions

**Environment.** The laptop's NVIDIA driver was found half-installed (broken kernel headers left modules unbuilt for the new kernel); reinstalling headers and reconfiguring fixed it. The install lives on the root filesystem (`/opt/isaac`) with Omniverse caches symlinked off `/home` by design.

**API version skew.** The installed Isaac Lab diverges from documentation in several places: `MeshConverterCfg` moved collision approximation into a schema object, `UsdFileCfg` dropped direct physics-material assignment (now bound as a separate material prim), and `pxr` is only importable inside the running app. Each was resolved by reading the actual source rather than guessing. The most consequential: this build's mesh converter silently emits pure geometry with no physics APIs, which produced a "no rigid body found" failure at spawn. Diagnosed with a USD-inspection probe; fixed with a small script that stamps rigid-body, mass, and convex-hull collision APIs onto the USD after every conversion.

**Process hygiene.** Isaac's headless shutdown frequently wedges, and suspended processes retain GPU memory; accumulated zombies eventually starved PhysX of VRAM (allocation failure, no physics scene). Resolved by killing suspended runs promptly; one-shot scripts are being patched to hard-exit. A deprecation warning on the wrench-application API is being tolerated deliberately: the recommended replacement changes where forces are applied, so migration is deferred until it can be done with regression re-runs of the verified probes rather than mid-investigation.

## Current focus and next steps

The immediate question is what arrests the roll. A force-threshold probe now runs independent trials at increasing global magnet strength (M100 → M800): if the detent completes at some strength, the wall is force scale and the multiplier quantifies it; if the module stays pinned at ~8° regardless, the wall is geometric — pointing at the chamfer/contact geometry, where the step-3 pitch observation already casts suspicion. The probe also records the maximum yaw reached, distinguishing an overshoot-and-fall-back (energy/damping signature) from never-exceeding. The outcome selects the mitigation: rescaling the magnet model, revising the chamfer geometry, or adjusting the actuation recipe (e.g., not fully de-energizing the departing latch). Once rotation is demonstrated and characterized, step 5 assembles the full vectorized RL environment — reward logic carried over from MuJoCo, with one deliberate design decision flagged from the MuJoCo post-mortem: the seed hold must not be silently overwritten by the policy every step.

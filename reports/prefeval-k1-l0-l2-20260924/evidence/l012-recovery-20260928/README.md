# Read-only candidate audit; C/R not launched here

The shared-disk audit in audit-before.py completed successfully on the initially idle l012 instance. Before launch, a second check found an unrelated B730 exposure512 controller had started and would use all four GPUs. The prepared launch-l012.py was NEVER executed. Do not run it.

C/R were instead resumed on the verified empty NORMAL instance prefeval-k1-h200x4-high-20260924. See ../high-recovery-20260928. The frozen pre-launch logs/receipts under the remote l012-recovery-20260928/before directory are included in that recovery archive.

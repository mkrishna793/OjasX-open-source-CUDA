From Corelib Require Extraction.
From Ojasx Require Import Product Matrix Nonlinear Optimizer Planner.
From Ojasx Require Import Contracts Structured RankSeven Foundation.
From Ojasx Require Import Compact Indexed.

(* Retain arbitrary-precision inductive integers and exact rationals.
   Do not use extraction overrides mapping Qc/Z/N to machine floats/ints. *)
Extraction Language OCaml.
Set Extraction Output Directory "extracted".
Extraction "ojasx_math.ml"
  product reference_gradient product_pullback fast_gradient
  squared_loss reference_loss_gradient fast_loss_gradient
  gradient_update reference_step fast_step iterate_steps safeguarded_step
  mm transpose planned_forward planned_reverse
  matrix_sgd planned_parameter_updates
  relu positive_mask relu_pullback packed_relu_pullback gate_columns
  native_work contracted_work native_tape contracted_tape
  native_schedule contracted_schedule schedule_work
  choose_contracted selected_work selected_tape.

Extraction "ojasx_foundation.ml"
  structured_push structured_pull softmax_pull structured_work explicit_dense_work
  seven conventional mul7 mul2 reverse7 block7 block_native block_at
  winograd mulw blockw reversew winograd_joint_work choose_winograd
  ordinary_joint_work seven_joint_work choose_seven selected_product.

Extraction "ojasx_refined.ml"
  compact_rms_pull compact_layer_pull gain_pull bias_pull
  gather scatter_sum.


type bool =
| True
| False

type nat =
| O
| S of nat

type ('a, 'b) prod =
| Pair of 'a * 'b

val fst : ('a1, 'a2) prod -> 'a1

val snd : ('a1, 'a2) prod -> 'a2

type 'a list =
| Nil
| Cons of 'a * 'a list

type comparison =
| Eq
| Lt
| Gt

val compOpp : comparison -> comparison

type sumbool =
| Left
| Right

val add : nat -> nat -> nat

type positive =
| XI of positive
| XO of positive
| XH

type n =
| N0
| Npos of positive

type z =
| Z0
| Zpos of positive
| Zneg of positive

module Pos :
 sig
  val succ : positive -> positive

  val add : positive -> positive -> positive

  val add_carry : positive -> positive -> positive

  val pred_double : positive -> positive

  type mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  val mul : positive -> positive -> positive

  val compare_cont : comparison -> positive -> positive -> comparison

  val compare : positive -> positive -> comparison

  val of_succ_nat : nat -> positive
 end

module Coq_Pos :
 sig
  val succ : positive -> positive

  val add : positive -> positive -> positive

  val add_carry : positive -> positive -> positive

  val pred_double : positive -> positive

  type mask = Pos.mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  val succ_double_mask : mask -> mask

  val double_mask : mask -> mask

  val double_pred_mask : positive -> mask

  val sub_mask : positive -> positive -> mask

  val sub_mask_carry : positive -> positive -> mask

  val sub : positive -> positive -> positive

  val mul : positive -> positive -> positive

  val compare_cont : comparison -> positive -> positive -> comparison

  val compare : positive -> positive -> comparison

  val size_nat : positive -> nat

  val ggcdn :
    nat -> positive -> positive -> (positive, (positive, positive) prod) prod

  val ggcd :
    positive -> positive -> (positive, (positive, positive) prod) prod
 end

module N :
 sig
  val compare : n -> n -> comparison

  val leb : n -> n -> bool

  val add : n -> n -> n

  val mul : n -> n -> n

  val of_nat : nat -> n
 end

module Z :
 sig
  val double : z -> z

  val succ_double : z -> z

  val pred_double : z -> z

  val pos_sub : positive -> positive -> z

  val add : z -> z -> z

  val opp : z -> z

  val mul : z -> z -> z

  val compare : z -> z -> comparison

  val to_pos : z -> positive

  val sgn : z -> z

  val abs : z -> z

  val ggcd : z -> z -> (z, (z, z) prod) prod
 end

val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list

val z_lt_dec : z -> z -> sumbool

val z_lt_ge_dec : z -> z -> sumbool

val z_lt_le_dec : z -> z -> sumbool

type q = { qnum : z; qden : positive }

val qplus : q -> q -> q

val qmult : q -> q -> q

val qopp : q -> q

val qinv : q -> q

val qlt_le_dec : q -> q -> sumbool

val qred : q -> q

type qc = q
  (* singleton inductive, whose constructor was Qcmake *)

val q2Qc : q -> qc

val qcplus : qc -> qc -> qc

val qcmult : qc -> qc -> qc

val qcopp : qc -> qc

val qcminus : qc -> qc -> qc

val qcinv : qc -> qc

val qcdiv : qc -> qc -> qc

val qclt_le_dec : qc -> qc -> sumbool

val product : qc list -> qc

val scale : qc -> qc list -> qc list

val reference_gradient : qc list -> qc list

val product_pullback : qc list -> qc -> (qc, qc list) prod

val fast_gradient : qc list -> qc list

val squared_loss : qc list -> qc -> qc

val reference_loss_gradient : qc list -> qc -> qc list

val fast_loss_gradient : qc list -> qc -> qc list

type matrix = nat -> nat -> qc

val qsum : nat -> (nat -> qc) -> qc

val mm : nat -> matrix -> matrix -> matrix

val transpose : matrix -> matrix

val gate : bool -> qc -> qc

val positive_mask : qc -> bool

val relu : qc -> qc

val relu_pullback : qc -> qc -> qc

val packed_relu_pullback : bool -> qc -> qc

val gate_columns : (nat -> bool) -> matrix -> matrix

val gradient_update : qc list -> qc list -> qc -> qc list

val reference_step : qc list -> qc -> qc -> qc list

val fast_step : qc list -> qc -> qc -> qc list

val iterate_steps : nat -> (qc list -> qc list) -> qc list -> qc list

val safeguarded_step : nat -> qc list -> qc -> qc -> qc list

val native_work : n -> n -> n -> n -> n

val contracted_work : n -> n -> n -> n -> n

type gemm_shape = ((n, n) prod, n) prod

val native_schedule : n -> n -> n -> n -> gemm_shape list

val contracted_schedule : n -> n -> n -> n -> gemm_shape list

val schedule_work : gemm_shape list -> n

val native_tape : n -> n -> n

val contracted_tape : n -> n -> n

val choose_contracted : n -> n -> n -> n -> bool

val selected_work : n -> n -> n -> n -> n

val selected_tape : n -> n -> n -> n -> n

val planned_forward :
  nat -> nat -> nat -> nat -> matrix -> matrix -> matrix -> matrix

val planned_reverse :
  nat -> nat -> nat -> nat -> matrix -> matrix -> matrix -> matrix ->
  ((matrix, matrix) prod, matrix) prod

val matrix_sgd : matrix -> matrix -> qc -> matrix

val planned_parameter_updates :
  nat -> nat -> nat -> nat -> matrix -> matrix -> matrix -> matrix -> qc ->
  (matrix, matrix) prod

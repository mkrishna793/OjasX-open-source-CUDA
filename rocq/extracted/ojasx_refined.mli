
type bool =
| True
| False

type nat =
| O
| S of nat

type ('a, 'b) prod =
| Pair of 'a * 'b

val snd : ('a1, 'a2) prod -> 'a2

type comparison =
| Eq
| Lt
| Gt

val add : nat -> nat -> nat

type positive =
| XI of positive
| XO of positive
| XH

type z =
| Z0
| Zpos of positive
| Zneg of positive

module Nat :
 sig
  val eqb : nat -> nat -> bool
 end

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

module Z :
 sig
  val double : z -> z

  val succ_double : z -> z

  val pred_double : z -> z

  val pos_sub : positive -> positive -> z

  val add : z -> z -> z

  val opp : z -> z

  val mul : z -> z -> z

  val to_pos : z -> positive

  val sgn : z -> z

  val abs : z -> z

  val ggcd : z -> z -> (z, (z, z) prod) prod
 end

type q = { qnum : z; qden : positive }

val qplus : q -> q -> q

val qmult : q -> q -> q

val qopp : q -> q

val qred : q -> q

type qc = q
  (* singleton inductive, whose constructor was Qcmake *)

val q2Qc : q -> qc

val qcplus : qc -> qc -> qc

val qcmult : qc -> qc -> qc

val qcopp : qc -> qc

val qcminus : qc -> qc -> qc

type matrix = nat -> nat -> qc

val qsum : nat -> (nat -> qc) -> qc

type vector = nat -> qc

val compact_rms_pull : nat -> qc -> qc -> vector -> vector -> nat -> qc

val compact_layer_pull : nat -> qc -> qc -> vector -> vector -> nat -> qc

val gain_pull : nat -> matrix -> matrix -> nat -> qc

val bias_pull : nat -> matrix -> nat -> qc

val delta : nat -> nat -> qc

val gather : (nat -> nat) -> vector -> vector

val scatter_sum : nat -> (nat -> nat) -> vector -> vector

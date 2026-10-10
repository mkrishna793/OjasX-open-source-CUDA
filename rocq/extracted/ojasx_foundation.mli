
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

val mm : nat -> matrix -> matrix -> matrix

type vector = nat -> qc

val factor_apply : nat -> nat -> matrix -> matrix -> vector -> vector

val structured_push :
  nat -> nat -> vector -> matrix -> matrix -> vector -> vector

val structured_pull :
  nat -> nat -> vector -> matrix -> matrix -> vector -> vector

val softmax_pull : nat -> vector -> vector -> vector

val structured_work : n -> n -> n

val explicit_dense_work : n -> n

type 't quad = { a00 : 't; a01 : 't; a10 : 't; a11 : 't }

val seven :
  ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> 'a1
  quad -> 'a1 quad -> 'a1 quad

val conventional :
  ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> 'a1 quad -> 'a1 quad -> 'a1
  quad

val mul7 : qc quad -> qc quad -> qc quad

val mul2 : qc quad -> qc quad -> qc quad

val trans2 : qc quad -> qc quad

val reverse7 : qc quad -> qc quad -> qc quad -> (qc quad, qc quad) prod

val matrix_add : matrix -> matrix -> matrix

val matrix_sub : matrix -> matrix -> matrix

val block7 : nat -> matrix quad -> matrix quad -> matrix quad

val block_native : nat -> matrix quad -> matrix quad -> matrix quad

val block_at : matrix quad -> nat -> nat -> qc quad

val ordinary_joint_work : n -> n -> n

val seven_joint_work : n -> n -> n

val choose_seven : n -> n -> n -> n -> bool

val selected_product : n -> n -> n -> n -> qc quad -> qc quad -> qc quad

val winograd :
  ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> 'a1
  quad -> 'a1 quad -> 'a1 quad

val mulw : qc quad -> qc quad -> qc quad

val blockw : nat -> matrix quad -> matrix quad -> matrix quad

val reversew : qc quad -> qc quad -> qc quad -> (qc quad, qc quad) prod

val winograd_joint_work : n -> n -> n

val choose_winograd : n -> n -> n -> n -> bool

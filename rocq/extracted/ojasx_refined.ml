
type bool =
| True
| False

type nat =
| O
| S of nat

type ('a, 'b) prod =
| Pair of 'a * 'b

(** val snd : ('a1, 'a2) prod -> 'a2 **)

let snd = function
| Pair (_, y) -> y

type comparison =
| Eq
| Lt
| Gt

module Coq__1 = struct
 (** val add : nat -> nat -> nat **)

 let rec add n m =
   match n with
   | O -> m
   | S p -> S (add p m)
end
include Coq__1

type positive =
| XI of positive
| XO of positive
| XH

type z =
| Z0
| Zpos of positive
| Zneg of positive

module Nat =
 struct
  (** val eqb : nat -> nat -> bool **)

  let rec eqb n m =
    match n with
    | O -> (match m with
            | O -> True
            | S _ -> False)
    | S n' -> (match m with
               | O -> False
               | S m' -> eqb n' m')
 end

module Pos =
 struct
  (** val succ : positive -> positive **)

  let rec succ = function
  | XI p -> XO (succ p)
  | XO p -> XI p
  | XH -> XO XH

  (** val add : positive -> positive -> positive **)

  let rec add x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XI (add p q0)
       | XO q0 -> XO (add p q0)
       | XH -> XI p)
    | XH -> (match y with
             | XI q0 -> XO (succ q0)
             | XO q0 -> XI q0
             | XH -> XO XH)

  (** val add_carry : positive -> positive -> positive **)

  and add_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XI (add_carry p q0)
       | XO q0 -> XO (add_carry p q0)
       | XH -> XI (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XH ->
      (match y with
       | XI q0 -> XI (succ q0)
       | XO q0 -> XO (succ q0)
       | XH -> XI XH)

  (** val pred_double : positive -> positive **)

  let rec pred_double = function
  | XI p -> XI (XO p)
  | XO p -> XI (pred_double p)
  | XH -> XH

  type mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  (** val mul : positive -> positive -> positive **)

  let rec mul x y =
    match x with
    | XI p -> add y (XO (mul p y))
    | XO p -> XO (mul p y)
    | XH -> y
 end

module Coq_Pos =
 struct
  (** val succ : positive -> positive **)

  let rec succ = function
  | XI p -> XO (succ p)
  | XO p -> XI p
  | XH -> XO XH

  (** val add : positive -> positive -> positive **)

  let rec add x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XI (add p q0)
       | XO q0 -> XO (add p q0)
       | XH -> XI p)
    | XH -> (match y with
             | XI q0 -> XO (succ q0)
             | XO q0 -> XI q0
             | XH -> XO XH)

  (** val add_carry : positive -> positive -> positive **)

  and add_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XI (add_carry p q0)
       | XO q0 -> XO (add_carry p q0)
       | XH -> XI (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XH ->
      (match y with
       | XI q0 -> XI (succ q0)
       | XO q0 -> XO (succ q0)
       | XH -> XI XH)

  (** val pred_double : positive -> positive **)

  let rec pred_double = function
  | XI p -> XI (XO p)
  | XO p -> XI (pred_double p)
  | XH -> XH

  type mask = Pos.mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  (** val succ_double_mask : mask -> mask **)

  let succ_double_mask = function
  | IsNul -> IsPos XH
  | IsPos p -> IsPos (XI p)
  | IsNeg -> IsNeg

  (** val double_mask : mask -> mask **)

  let double_mask = function
  | IsPos p -> IsPos (XO p)
  | x0 -> x0

  (** val double_pred_mask : positive -> mask **)

  let double_pred_mask = function
  | XI p -> IsPos (XO (XO p))
  | XO p -> IsPos (XO (pred_double p))
  | XH -> IsNul

  (** val sub_mask : positive -> positive -> mask **)

  let rec sub_mask x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> double_mask (sub_mask p q0)
       | XO q0 -> succ_double_mask (sub_mask p q0)
       | XH -> IsPos (XO p))
    | XO p ->
      (match y with
       | XI q0 -> succ_double_mask (sub_mask_carry p q0)
       | XO q0 -> double_mask (sub_mask p q0)
       | XH -> IsPos (pred_double p))
    | XH -> (match y with
             | XH -> IsNul
             | _ -> IsNeg)

  (** val sub_mask_carry : positive -> positive -> mask **)

  and sub_mask_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> succ_double_mask (sub_mask_carry p q0)
       | XO q0 -> double_mask (sub_mask p q0)
       | XH -> IsPos (pred_double p))
    | XO p ->
      (match y with
       | XI q0 -> double_mask (sub_mask_carry p q0)
       | XO q0 -> succ_double_mask (sub_mask_carry p q0)
       | XH -> double_pred_mask p)
    | XH -> IsNeg

  (** val sub : positive -> positive -> positive **)

  let sub x y =
    match sub_mask x y with
    | IsPos z0 -> z0
    | _ -> XH

  (** val mul : positive -> positive -> positive **)

  let rec mul x y =
    match x with
    | XI p -> add y (XO (mul p y))
    | XO p -> XO (mul p y)
    | XH -> y

  (** val compare_cont : comparison -> positive -> positive -> comparison **)

  let rec compare_cont r x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> compare_cont r p q0
       | XO q0 -> compare_cont Gt p q0
       | XH -> Gt)
    | XO p ->
      (match y with
       | XI q0 -> compare_cont Lt p q0
       | XO q0 -> compare_cont r p q0
       | XH -> Gt)
    | XH -> (match y with
             | XH -> r
             | _ -> Lt)

  (** val compare : positive -> positive -> comparison **)

  let compare =
    compare_cont Eq

  (** val size_nat : positive -> nat **)

  let rec size_nat = function
  | XI p0 -> S (size_nat p0)
  | XO p0 -> S (size_nat p0)
  | XH -> S O

  (** val ggcdn :
      nat -> positive -> positive -> (positive, (positive, positive) prod)
      prod **)

  let rec ggcdn n a b =
    match n with
    | O -> Pair (XH, (Pair (a, b)))
    | S n0 ->
      (match a with
       | XI a' ->
         (match b with
          | XI b' ->
            (match compare a' b' with
             | Eq -> Pair (a, (Pair (XH, XH)))
             | Lt ->
               let Pair (g, p) = ggcdn n0 (sub b' a') a in
               let Pair (ba, aa) = p in
               Pair (g, (Pair (aa, (add aa (XO ba)))))
             | Gt ->
               let Pair (g, p) = ggcdn n0 (sub a' b') b in
               let Pair (ab, bb) = p in
               Pair (g, (Pair ((add bb (XO ab)), bb))))
          | XO b0 ->
            let Pair (g, p) = ggcdn n0 a b0 in
            let Pair (aa, bb) = p in Pair (g, (Pair (aa, (XO bb))))
          | XH -> Pair (XH, (Pair (a, XH))))
       | XO a0 ->
         (match b with
          | XI _ ->
            let Pair (g, p) = ggcdn n0 a0 b in
            let Pair (aa, bb) = p in Pair (g, (Pair ((XO aa), bb)))
          | XO b0 -> let Pair (g, p) = ggcdn n0 a0 b0 in Pair ((XO g), p)
          | XH -> Pair (XH, (Pair (a, XH))))
       | XH -> Pair (XH, (Pair (XH, b))))

  (** val ggcd :
      positive -> positive -> (positive, (positive, positive) prod) prod **)

  let ggcd a b =
    ggcdn (Coq__1.add (size_nat a) (size_nat b)) a b
 end

module Z =
 struct
  (** val double : z -> z **)

  let double = function
  | Z0 -> Z0
  | Zpos p -> Zpos (XO p)
  | Zneg p -> Zneg (XO p)

  (** val succ_double : z -> z **)

  let succ_double = function
  | Z0 -> Zpos XH
  | Zpos p -> Zpos (XI p)
  | Zneg p -> Zneg (Pos.pred_double p)

  (** val pred_double : z -> z **)

  let pred_double = function
  | Z0 -> Zneg XH
  | Zpos p -> Zpos (Pos.pred_double p)
  | Zneg p -> Zneg (XI p)

  (** val pos_sub : positive -> positive -> z **)

  let rec pos_sub x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> double (pos_sub p q0)
       | XO q0 -> succ_double (pos_sub p q0)
       | XH -> Zpos (XO p))
    | XO p ->
      (match y with
       | XI q0 -> pred_double (pos_sub p q0)
       | XO q0 -> double (pos_sub p q0)
       | XH -> Zpos (Pos.pred_double p))
    | XH ->
      (match y with
       | XI q0 -> Zneg (XO q0)
       | XO q0 -> Zneg (Pos.pred_double q0)
       | XH -> Z0)

  (** val add : z -> z -> z **)

  let add x y =
    match x with
    | Z0 -> y
    | Zpos x' ->
      (match y with
       | Z0 -> x
       | Zpos y' -> Zpos (Pos.add x' y')
       | Zneg y' -> pos_sub x' y')
    | Zneg x' ->
      (match y with
       | Z0 -> x
       | Zpos y' -> pos_sub y' x'
       | Zneg y' -> Zneg (Pos.add x' y'))

  (** val opp : z -> z **)

  let opp = function
  | Z0 -> Z0
  | Zpos x0 -> Zneg x0
  | Zneg x0 -> Zpos x0

  (** val mul : z -> z -> z **)

  let mul x y =
    match x with
    | Z0 -> Z0
    | Zpos x' ->
      (match y with
       | Z0 -> Z0
       | Zpos y' -> Zpos (Pos.mul x' y')
       | Zneg y' -> Zneg (Pos.mul x' y'))
    | Zneg x' ->
      (match y with
       | Z0 -> Z0
       | Zpos y' -> Zneg (Pos.mul x' y')
       | Zneg y' -> Zpos (Pos.mul x' y'))

  (** val to_pos : z -> positive **)

  let to_pos = function
  | Zpos p -> p
  | _ -> XH

  (** val sgn : z -> z **)

  let sgn = function
  | Z0 -> Z0
  | Zpos _ -> Zpos XH
  | Zneg _ -> Zneg XH

  (** val abs : z -> z **)

  let abs = function
  | Zneg p -> Zpos p
  | x -> x

  (** val ggcd : z -> z -> (z, (z, z) prod) prod **)

  let ggcd a b =
    match a with
    | Z0 -> Pair ((abs b), (Pair (Z0, (sgn b))))
    | Zpos a0 ->
      (match b with
       | Z0 -> Pair ((abs a), (Pair ((sgn a), Z0)))
       | Zpos b0 ->
         let Pair (g, p) = Coq_Pos.ggcd a0 b0 in
         let Pair (aa, bb) = p in
         Pair ((Zpos g), (Pair ((Zpos aa), (Zpos bb))))
       | Zneg b0 ->
         let Pair (g, p) = Coq_Pos.ggcd a0 b0 in
         let Pair (aa, bb) = p in
         Pair ((Zpos g), (Pair ((Zpos aa), (Zneg bb)))))
    | Zneg a0 ->
      (match b with
       | Z0 -> Pair ((abs a), (Pair ((sgn a), Z0)))
       | Zpos b0 ->
         let Pair (g, p) = Coq_Pos.ggcd a0 b0 in
         let Pair (aa, bb) = p in
         Pair ((Zpos g), (Pair ((Zneg aa), (Zpos bb))))
       | Zneg b0 ->
         let Pair (g, p) = Coq_Pos.ggcd a0 b0 in
         let Pair (aa, bb) = p in
         Pair ((Zpos g), (Pair ((Zneg aa), (Zneg bb)))))
 end

type q = { qnum : z; qden : positive }

(** val qplus : q -> q -> q **)

let qplus x y =
  { qnum = (Z.add (Z.mul x.qnum (Zpos y.qden)) (Z.mul y.qnum (Zpos x.qden)));
    qden = (Coq_Pos.mul x.qden y.qden) }

(** val qmult : q -> q -> q **)

let qmult x y =
  { qnum = (Z.mul x.qnum y.qnum); qden = (Coq_Pos.mul x.qden y.qden) }

(** val qopp : q -> q **)

let qopp x =
  { qnum = (Z.opp x.qnum); qden = x.qden }

(** val qred : q -> q **)

let qred q0 =
  let { qnum = q1; qden = q2 } = q0 in
  let Pair (r1, r2) = snd (Z.ggcd q1 (Zpos q2)) in
  { qnum = r1; qden = (Z.to_pos r2) }

type qc = q
  (* singleton inductive, whose constructor was Qcmake *)

(** val q2Qc : q -> qc **)

let q2Qc =
  qred

(** val qcplus : qc -> qc -> qc **)

let qcplus x y =
  q2Qc (qplus x y)

(** val qcmult : qc -> qc -> qc **)

let qcmult x y =
  q2Qc (qmult x y)

(** val qcopp : qc -> qc **)

let qcopp x =
  q2Qc (qopp x)

(** val qcminus : qc -> qc -> qc **)

let qcminus x y =
  qcplus x (qcopp y)

type matrix = nat -> nat -> qc

(** val qsum : nat -> (nat -> qc) -> qc **)

let rec qsum n f =
  match n with
  | O -> q2Qc { qnum = Z0; qden = XH }
  | S k -> qcplus (qsum k f) (f k)

type vector = nat -> qc

(** val compact_rms_pull :
    nat -> qc -> qc -> vector -> vector -> nat -> qc **)

let compact_rms_pull n r invn z0 q0 i =
  qcmult r
    (qcminus (q0 i)
      (qcmult (qcmult (z0 i) invn) (qsum n (fun j -> qcmult (z0 j) (q0 j)))))

(** val compact_layer_pull :
    nat -> qc -> qc -> vector -> vector -> nat -> qc **)

let compact_layer_pull n r invn z0 q0 i =
  qcmult r
    (qcminus (qcminus (q0 i) (qcmult invn (qsum n q0)))
      (qcmult (qcmult (z0 i) invn) (qsum n (fun j -> qcmult (z0 j) (q0 j)))))

(** val gain_pull : nat -> matrix -> matrix -> nat -> qc **)

let gain_pull m z0 g j =
  qsum m (fun i -> qcmult (z0 i j) (g i j))

(** val bias_pull : nat -> matrix -> nat -> qc **)

let bias_pull m g j =
  qsum m (fun i -> g i j)

(** val delta : nat -> nat -> qc **)

let delta i j =
  match Nat.eqb i j with
  | True -> q2Qc { qnum = (Zpos XH); qden = XH }
  | False -> q2Qc { qnum = Z0; qden = XH }

(** val gather : (nat -> nat) -> vector -> vector **)

let gather index x j =
  x (index j)

(** val scatter_sum : nat -> (nat -> nat) -> vector -> vector **)

let scatter_sum k index g i =
  qsum k (fun j -> qcmult (g j) (delta (index j) i))

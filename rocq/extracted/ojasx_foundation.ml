
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

 let rec add n0 m =
   match n0 with
   | O -> m
   | S p -> S (add p m)
end
include Coq__1

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

  let rec ggcdn n0 a b =
    match n0 with
    | O -> Pair (XH, (Pair (a, b)))
    | S n1 ->
      (match a with
       | XI a' ->
         (match b with
          | XI b' ->
            (match compare a' b' with
             | Eq -> Pair (a, (Pair (XH, XH)))
             | Lt ->
               let Pair (g, p) = ggcdn n1 (sub b' a') a in
               let Pair (ba, aa) = p in
               Pair (g, (Pair (aa, (add aa (XO ba)))))
             | Gt ->
               let Pair (g, p) = ggcdn n1 (sub a' b') b in
               let Pair (ab, bb) = p in
               Pair (g, (Pair ((add bb (XO ab)), bb))))
          | XO b0 ->
            let Pair (g, p) = ggcdn n1 a b0 in
            let Pair (aa, bb) = p in Pair (g, (Pair (aa, (XO bb))))
          | XH -> Pair (XH, (Pair (a, XH))))
       | XO a0 ->
         (match b with
          | XI _ ->
            let Pair (g, p) = ggcdn n1 a0 b in
            let Pair (aa, bb) = p in Pair (g, (Pair ((XO aa), bb)))
          | XO b0 -> let Pair (g, p) = ggcdn n1 a0 b0 in Pair ((XO g), p)
          | XH -> Pair (XH, (Pair (a, XH))))
       | XH -> Pair (XH, (Pair (XH, b))))

  (** val ggcd :
      positive -> positive -> (positive, (positive, positive) prod) prod **)

  let ggcd a b =
    ggcdn (Coq__1.add (size_nat a) (size_nat b)) a b
 end

module N =
 struct
  (** val compare : n -> n -> comparison **)

  let compare n0 m =
    match n0 with
    | N0 -> (match m with
             | N0 -> Eq
             | Npos _ -> Lt)
    | Npos n' -> (match m with
                  | N0 -> Gt
                  | Npos m' -> Pos.compare n' m')

  (** val leb : n -> n -> bool **)

  let leb x y =
    match compare x y with
    | Gt -> False
    | _ -> True

  (** val add : n -> n -> n **)

  let add n0 m =
    match n0 with
    | N0 -> m
    | Npos p -> (match m with
                 | N0 -> n0
                 | Npos q0 -> Npos (Coq_Pos.add p q0))

  (** val mul : n -> n -> n **)

  let mul n0 m =
    match n0 with
    | N0 -> N0
    | Npos p -> (match m with
                 | N0 -> N0
                 | Npos q0 -> Npos (Coq_Pos.mul p q0))
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

let rec qsum n0 f =
  match n0 with
  | O -> q2Qc { qnum = Z0; qden = XH }
  | S k -> qcplus (qsum k f) (f k)

(** val mm : nat -> matrix -> matrix -> matrix **)

let mm k a b i j =
  qsum k (fun h -> qcmult (a i h) (b h j))

type vector = nat -> qc

(** val factor_apply : nat -> nat -> matrix -> matrix -> vector -> vector **)

let factor_apply n0 r u v x i =
  qsum r (fun h -> qcmult (u i h) (qsum n0 (fun j -> qcmult (v j h) (x j))))

(** val structured_push :
    nat -> nat -> vector -> matrix -> matrix -> vector -> vector **)

let structured_push n0 r d u v x i =
  qcplus (qcmult (d i) (x i)) (factor_apply n0 r u v x i)

(** val structured_pull :
    nat -> nat -> vector -> matrix -> matrix -> vector -> vector **)

let structured_pull n0 r d u v g =
  structured_push n0 r d v u g

(** val softmax_pull : nat -> vector -> vector -> vector **)

let softmax_pull n0 p g i =
  qcmult (p i) (qcminus (g i) (qsum n0 (fun j -> qcmult (p j) (g j))))

(** val structured_work : n -> n -> n **)

let structured_work n0 r =
  N.add (N.mul (N.mul (Npos (XO (XO XH))) n0) r) (N.mul (Npos (XO XH)) n0)

(** val explicit_dense_work : n -> n **)

let explicit_dense_work n0 =
  N.mul (N.mul (Npos (XO XH)) n0) n0

type 't quad = { a00 : 't; a01 : 't; a10 : 't; a11 : 't }

(** val seven :
    ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> 'a1
    quad -> 'a1 quad -> 'a1 quad **)

let seven add0 sub0 mul0 a b =
  let p1 = mul0 (add0 a.a00 a.a11) (add0 b.a00 b.a11) in
  let p2 = mul0 (add0 a.a10 a.a11) b.a00 in
  let p3 = mul0 a.a00 (sub0 b.a01 b.a11) in
  let p4 = mul0 a.a11 (sub0 b.a10 b.a00) in
  let p5 = mul0 (add0 a.a00 a.a01) b.a11 in
  let p6 = mul0 (sub0 a.a10 a.a00) (add0 b.a00 b.a01) in
  let p7 = mul0 (sub0 a.a01 a.a11) (add0 b.a10 b.a11) in
  { a00 = (add0 (sub0 (add0 p1 p4) p5) p7); a01 = (add0 p3 p5); a10 =
  (add0 p2 p4); a11 = (add0 (add0 (sub0 p1 p2) p3) p6) }

(** val conventional :
    ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> 'a1 quad -> 'a1 quad -> 'a1
    quad **)

let conventional add0 mul0 a b =
  { a00 = (add0 (mul0 a.a00 b.a00) (mul0 a.a01 b.a10)); a01 =
    (add0 (mul0 a.a00 b.a01) (mul0 a.a01 b.a11)); a10 =
    (add0 (mul0 a.a10 b.a00) (mul0 a.a11 b.a10)); a11 =
    (add0 (mul0 a.a10 b.a01) (mul0 a.a11 b.a11)) }

(** val mul7 : qc quad -> qc quad -> qc quad **)

let mul7 =
  seven qcplus qcminus qcmult

(** val mul2 : qc quad -> qc quad -> qc quad **)

let mul2 =
  conventional qcplus qcmult

(** val trans2 : qc quad -> qc quad **)

let trans2 a =
  { a00 = a.a00; a01 = a.a10; a10 = a.a01; a11 = a.a11 }

(** val reverse7 :
    qc quad -> qc quad -> qc quad -> (qc quad, qc quad) prod **)

let reverse7 a b g =
  Pair ((mul7 g (trans2 b)), (mul7 (trans2 a) g))

(** val matrix_add : matrix -> matrix -> matrix **)

let matrix_add a b i j =
  qcplus (a i j) (b i j)

(** val matrix_sub : matrix -> matrix -> matrix **)

let matrix_sub a b i j =
  qcminus (a i j) (b i j)

(** val block7 : nat -> matrix quad -> matrix quad -> matrix quad **)

let block7 k =
  seven matrix_add matrix_sub (mm k)

(** val block_native : nat -> matrix quad -> matrix quad -> matrix quad **)

let block_native k =
  conventional matrix_add (mm k)

(** val block_at : matrix quad -> nat -> nat -> qc quad **)

let block_at a i j =
  { a00 = (a.a00 i j); a01 = (a.a01 i j); a10 = (a.a10 i j); a11 =
    (a.a11 i j) }

(** val ordinary_joint_work : n -> n -> n **)

let ordinary_joint_work mult add0 =
  N.add (N.mul (Npos (XO (XO (XO (XI XH))))) mult)
    (N.mul (Npos (XO (XO (XI XH)))) add0)

(** val seven_joint_work : n -> n -> n **)

let seven_joint_work mult add0 =
  N.add (N.mul (Npos (XI (XO (XI (XO XH))))) mult)
    (N.mul (Npos (XO (XI (XI (XO (XI XH)))))) add0)

(** val choose_seven : n -> n -> n -> n -> bool **)

let choose_seven mult add0 scratch budget =
  match N.leb (seven_joint_work mult add0) (ordinary_joint_work mult add0) with
  | True -> N.leb scratch budget
  | False -> False

(** val selected_product :
    n -> n -> n -> n -> qc quad -> qc quad -> qc quad **)

let selected_product mult add0 scratch budget a b =
  match choose_seven mult add0 scratch budget with
  | True -> mul7 a b
  | False -> mul2 a b

(** val winograd :
    ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> 'a1) -> 'a1
    quad -> 'a1 quad -> 'a1 quad **)

let winograd add0 sub0 mul0 a b =
  let s1 = add0 a.a10 a.a11 in
  let s2 = sub0 s1 a.a00 in
  let s3 = sub0 a.a00 a.a10 in
  let s4 = sub0 a.a01 s2 in
  let t1 = sub0 b.a01 b.a00 in
  let t2 = sub0 b.a11 t1 in
  let t3 = sub0 b.a11 b.a01 in
  let t4 = sub0 t2 b.a10 in
  let p1 = mul0 a.a00 b.a00 in
  let p2 = mul0 a.a01 b.a10 in
  let p3 = mul0 s4 b.a11 in
  let p4 = mul0 a.a11 t4 in
  let p5 = mul0 s1 t1 in
  let p6 = mul0 s2 t2 in
  let p7 = mul0 s3 t3 in
  let u1 = add0 p1 p6 in
  let u2 = add0 u1 p7 in
  let u3 = add0 u1 p5 in
  { a00 = (add0 p1 p2); a01 = (add0 u3 p3); a10 = (sub0 u2 p4); a11 =
  (add0 u2 p5) }

(** val mulw : qc quad -> qc quad -> qc quad **)

let mulw =
  winograd qcplus qcminus qcmult

(** val blockw : nat -> matrix quad -> matrix quad -> matrix quad **)

let blockw k =
  winograd matrix_add matrix_sub (mm k)

(** val reversew :
    qc quad -> qc quad -> qc quad -> (qc quad, qc quad) prod **)

let reversew a b g =
  Pair ((mulw g (trans2 b)), (mulw (trans2 a) g))

(** val winograd_joint_work : n -> n -> n **)

let winograd_joint_work mult add0 =
  N.add (N.mul (Npos (XI (XO (XI (XO XH))))) mult)
    (N.mul (Npos (XI (XO (XI (XI (XO XH)))))) add0)

(** val choose_winograd : n -> n -> n -> n -> bool **)

let choose_winograd mult add0 scratch budget =
  match N.leb (winograd_joint_work mult add0) (ordinary_joint_work mult add0) with
  | True -> N.leb scratch budget
  | False -> False

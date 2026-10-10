
type bool =
| True
| False

type nat =
| O
| S of nat

type ('a, 'b) prod =
| Pair of 'a * 'b

(** val fst : ('a1, 'a2) prod -> 'a1 **)

let fst = function
| Pair (x, _) -> x

(** val snd : ('a1, 'a2) prod -> 'a2 **)

let snd = function
| Pair (_, y) -> y

type 'a list =
| Nil
| Cons of 'a * 'a list

type comparison =
| Eq
| Lt
| Gt

(** val compOpp : comparison -> comparison **)

let compOpp = function
| Eq -> Eq
| Lt -> Gt
| Gt -> Lt

type sumbool =
| Left
| Right

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

  (** val of_succ_nat : nat -> positive **)

  let rec of_succ_nat = function
  | O -> XH
  | S x -> succ (of_succ_nat x)
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

  (** val of_nat : nat -> n **)

  let of_nat = function
  | O -> N0
  | S n' -> Npos (Pos.of_succ_nat n')
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

  (** val compare : z -> z -> comparison **)

  let compare x y =
    match x with
    | Z0 -> (match y with
             | Z0 -> Eq
             | Zpos _ -> Lt
             | Zneg _ -> Gt)
    | Zpos x' -> (match y with
                  | Zpos y' -> Pos.compare x' y'
                  | _ -> Gt)
    | Zneg x' ->
      (match y with
       | Zneg y' -> compOpp (Pos.compare x' y')
       | _ -> Lt)

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

(** val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list **)

let rec map f = function
| Nil -> Nil
| Cons (a, l0) -> Cons ((f a), (map f l0))

(** val z_lt_dec : z -> z -> sumbool **)

let z_lt_dec x y =
  match Z.compare x y with
  | Lt -> Left
  | _ -> Right

(** val z_lt_ge_dec : z -> z -> sumbool **)

let z_lt_ge_dec =
  z_lt_dec

(** val z_lt_le_dec : z -> z -> sumbool **)

let z_lt_le_dec =
  z_lt_ge_dec

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

(** val qinv : q -> q **)

let qinv x =
  match x.qnum with
  | Z0 -> { qnum = Z0; qden = XH }
  | Zpos p -> { qnum = (Zpos x.qden); qden = p }
  | Zneg p -> { qnum = (Zneg x.qden); qden = p }

(** val qlt_le_dec : q -> q -> sumbool **)

let qlt_le_dec x y =
  z_lt_le_dec (Z.mul x.qnum (Zpos y.qden)) (Z.mul y.qnum (Zpos x.qden))

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

(** val qcinv : qc -> qc **)

let qcinv x =
  q2Qc (qinv x)

(** val qcdiv : qc -> qc -> qc **)

let qcdiv x y =
  qcmult x (qcinv y)

(** val qclt_le_dec : qc -> qc -> sumbool **)

let qclt_le_dec =
  qlt_le_dec

(** val product : qc list -> qc **)

let rec product = function
| Nil -> q2Qc { qnum = (Zpos XH); qden = XH }
| Cons (x, tail) -> qcmult x (product tail)

(** val scale : qc -> qc list -> qc list **)

let scale p xs =
  map (fun x -> qcmult p x) xs

(** val reference_gradient : qc list -> qc list **)

let rec reference_gradient = function
| Nil -> Nil
| Cons (x, tail) -> Cons ((product tail), (scale x (reference_gradient tail)))

(** val product_pullback : qc list -> qc -> (qc, qc list) prod **)

let rec product_pullback xs prefix =
  match xs with
  | Nil -> Pair ((q2Qc { qnum = (Zpos XH); qden = XH }), Nil)
  | Cons (x, tail) ->
    let Pair (suffix, gradients) = product_pullback tail (qcmult prefix x) in
    Pair ((qcmult x suffix), (Cons ((qcmult prefix suffix), gradients)))

(** val fast_gradient : qc list -> qc list **)

let fast_gradient xs =
  snd (product_pullback xs (q2Qc { qnum = (Zpos XH); qden = XH }))

(** val squared_loss : qc list -> qc -> qc **)

let squared_loss xs target =
  qcmult (qcminus (product xs) target) (qcminus (product xs) target)

(** val reference_loss_gradient : qc list -> qc -> qc list **)

let reference_loss_gradient xs target =
  scale
    (qcmult
      (qcplus (q2Qc { qnum = (Zpos XH); qden = XH })
        (q2Qc { qnum = (Zpos XH); qden = XH }))
      (qcminus (product xs) target))
    (reference_gradient xs)

(** val fast_loss_gradient : qc list -> qc -> qc list **)

let fast_loss_gradient xs target =
  let Pair (value, grads) =
    product_pullback xs (q2Qc { qnum = (Zpos XH); qden = XH })
  in
  scale
    (qcmult
      (qcplus (q2Qc { qnum = (Zpos XH); qden = XH })
        (q2Qc { qnum = (Zpos XH); qden = XH }))
      (qcminus value target))
    grads

type matrix = nat -> nat -> qc

(** val qsum : nat -> (nat -> qc) -> qc **)

let rec qsum n0 f =
  match n0 with
  | O -> q2Qc { qnum = Z0; qden = XH }
  | S k -> qcplus (qsum k f) (f k)

(** val mm : nat -> matrix -> matrix -> matrix **)

let mm k a b i j =
  qsum k (fun h -> qcmult (a i h) (b h j))

(** val transpose : matrix -> matrix **)

let transpose a i j =
  a j i

(** val gate : bool -> qc -> qc **)

let gate active x =
  match active with
  | True -> x
  | False -> q2Qc { qnum = Z0; qden = XH }

(** val positive_mask : qc -> bool **)

let positive_mask x =
  match qclt_le_dec (q2Qc { qnum = Z0; qden = XH }) x with
  | Left -> True
  | Right -> False

(** val relu : qc -> qc **)

let relu x =
  gate (positive_mask x) x

(** val relu_pullback : qc -> qc -> qc **)

let relu_pullback x upstream =
  gate (positive_mask x) upstream

(** val packed_relu_pullback : bool -> qc -> qc **)

let packed_relu_pullback =
  gate

(** val gate_columns : (nat -> bool) -> matrix -> matrix **)

let gate_columns mask0 a i j =
  gate (mask0 j) (a i j)

(** val gradient_update : qc list -> qc list -> qc -> qc list **)

let rec gradient_update xs gs eta =
  match xs with
  | Nil -> Nil
  | Cons (x, xt) ->
    (match gs with
     | Nil -> Nil
     | Cons (g, gt) ->
       Cons ((qcminus x (qcmult eta g)), (gradient_update xt gt eta)))

(** val reference_step : qc list -> qc -> qc -> qc list **)

let reference_step xs target eta =
  gradient_update xs (reference_loss_gradient xs target) eta

(** val fast_step : qc list -> qc -> qc -> qc list **)

let fast_step xs target eta =
  gradient_update xs (fast_loss_gradient xs target) eta

(** val iterate_steps : nat -> (qc list -> qc list) -> qc list -> qc list **)

let rec iterate_steps fuel step xs =
  match fuel with
  | O -> xs
  | S n0 -> iterate_steps n0 step (step xs)

(** val safeguarded_step : nat -> qc list -> qc -> qc -> qc list **)

let rec safeguarded_step fuel xs target eta =
  match fuel with
  | O -> xs
  | S n0 ->
    let candidate = fast_step xs target eta in
    (match qclt_le_dec (squared_loss xs target)
             (squared_loss candidate target) with
     | Left ->
       safeguarded_step n0 xs target
         (qcdiv eta
           (qcplus (q2Qc { qnum = (Zpos XH); qden = XH })
             (q2Qc { qnum = (Zpos XH); qden = XH })))
     | Right -> candidate)

(** val native_work : n -> n -> n -> n -> n **)

let native_work m k n0 p =
  N.add (N.mul (N.mul (N.mul (Npos (XO (XI XH))) m) k) n0)
    (N.mul (N.mul (N.mul (Npos (XO (XI XH))) m) n0) p)

(** val contracted_work : n -> n -> n -> n -> n **)

let contracted_work m k n0 p =
  N.mul (N.mul (N.mul (Npos (XO (XI XH))) p) k) (N.add m n0)

type gemm_shape = ((n, n) prod, n) prod

(** val native_schedule : n -> n -> n -> n -> gemm_shape list **)

let native_schedule m k n0 p =
  Cons ((Pair ((Pair (m, k)), n0)), (Cons ((Pair ((Pair (m, n0)), p)), (Cons
    ((Pair ((Pair (m, p)), n0)), (Cons ((Pair ((Pair (m, n0)), k)), (Cons
    ((Pair ((Pair (k, m)), n0)), (Cons ((Pair ((Pair (n0, m)), p)),
    Nil)))))))))))

(** val contracted_schedule : n -> n -> n -> n -> gemm_shape list **)

let contracted_schedule m k n0 p =
  Cons ((Pair ((Pair (k, n0)), p)), (Cons ((Pair ((Pair (m, k)), p)), (Cons
    ((Pair ((Pair (m, p)), k)), (Cons ((Pair ((Pair (k, m)), p)), (Cons
    ((Pair ((Pair (k, p)), n0)), (Cons ((Pair ((Pair (n0, k)), p)),
    Nil)))))))))))

(** val schedule_work : gemm_shape list -> n **)

let rec schedule_work = function
| Nil -> N0
| Cons (g, rest) ->
  let Pair (p, c) = g in
  let Pair (r, h) = p in
  N.add (N.mul (N.mul (N.mul (Npos (XO XH)) r) h) c) (schedule_work rest)

(** val native_tape : n -> n -> n **)

let native_tape =
  N.mul

(** val contracted_tape : n -> n -> n **)

let contracted_tape =
  N.mul

(** val choose_contracted : n -> n -> n -> n -> bool **)

let choose_contracted m k n0 p =
  match N.leb (contracted_work m k n0 p) (native_work m k n0 p) with
  | True -> N.leb (contracted_tape k p) (native_tape m n0)
  | False -> False

(** val selected_work : n -> n -> n -> n -> n **)

let selected_work m k n0 p =
  match choose_contracted m k n0 p with
  | True -> contracted_work m k n0 p
  | False -> native_work m k n0 p

(** val selected_tape : n -> n -> n -> n -> n **)

let selected_tape m k n0 p =
  match choose_contracted m k n0 p with
  | True -> contracted_tape k p
  | False -> native_tape m n0

(** val planned_forward :
    nat -> nat -> nat -> nat -> matrix -> matrix -> matrix -> matrix **)

let planned_forward m k n0 p x a b =
  match choose_contracted (N.of_nat m) (N.of_nat k) (N.of_nat n0) (N.of_nat p) with
  | True -> mm k x (mm n0 a b)
  | False -> mm n0 (mm k x a) b

(** val planned_reverse :
    nat -> nat -> nat -> nat -> matrix -> matrix -> matrix -> matrix ->
    ((matrix, matrix) prod, matrix) prod **)

let planned_reverse m k n0 p x a b g =
  match choose_contracted (N.of_nat m) (N.of_nat k) (N.of_nat n0) (N.of_nat p) with
  | True ->
    let core = mm m (transpose x) g in
    Pair ((Pair ((mm p g (transpose (mm n0 a b))),
    (mm p core (transpose b)))), (mm k (transpose a) core))
  | False ->
    Pair ((Pair ((mm n0 (mm p g (transpose b)) (transpose a)),
      (mm m (transpose x) (mm p g (transpose b))))),
      (mm m (transpose (mm k x a)) g))

(** val matrix_sgd : matrix -> matrix -> qc -> matrix **)

let matrix_sgd w dw eta i j =
  qcminus (w i j) (qcmult eta (dw i j))

(** val planned_parameter_updates :
    nat -> nat -> nat -> nat -> matrix -> matrix -> matrix -> matrix -> qc ->
    (matrix, matrix) prod **)

let planned_parameter_updates m k n0 p x a b g eta =
  Pair ((matrix_sgd a (snd (fst (planned_reverse m k n0 p x a b g))) eta),
    (matrix_sgd b (snd (planned_reverse m k n0 p x a b g)) eta))

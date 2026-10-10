(* Hand-written boundary adapter; tested, not formally verified.
   Zarith is used only for decimal codecs. The engine retains extracted
   arbitrary-precision integers and canonical exact rationals. *)
module E = Ojasx_math

let rec pos_of_z x =
  if Z.equal x Z.one then E.XH
  else if Z.testbit x 0 then E.XI (pos_of_z (Z.shift_right x 1))
  else E.XO (pos_of_z (Z.shift_right x 1))
let rec z_of_pos = function
  | E.XH -> Z.one
  | E.XI p -> Z.succ (Z.shift_left (z_of_pos p) 1)
  | E.XO p -> Z.shift_left (z_of_pos p) 1
let e_z x = match Z.sign x with
  | 0 -> E.Z0 | 1 -> E.Zpos (pos_of_z x) | _ -> E.Zneg (pos_of_z (Z.neg x))
let of_e_z = function E.Z0 -> Z.zero | E.Zpos p -> z_of_pos p | E.Zneg p -> Z.neg (z_of_pos p)
let e_n s = let x = Z.of_string s in
  if Z.sign x < 0 then invalid_arg "negative dimension"
  else if Z.equal x Z.zero then E.N0 else E.Npos (pos_of_z x)
let of_e_n = function E.N0 -> "0" | E.Npos p -> Z.to_string (z_of_pos p)
let q_of_s s =
  let num, den = match String.split_on_char '/' s with
    | [n] -> Z.of_string n, Z.one
    | [n;d] -> Z.of_string n, Z.of_string d
    | _ -> invalid_arg "rational syntax" in
  if Z.sign den <= 0 then invalid_arg "denominator must be positive";
  E.q2Qc {E.qnum=e_z num; qden=pos_of_z den}
let s_of_q q = Z.to_string (of_e_z q.E.qnum) ^ "/" ^ Z.to_string (z_of_pos q.E.qden)
let rec nat_of_int n = if n=0 then E.O else E.S (nat_of_int (n-1))
let rec int_of_nat = function E.O -> 0 | E.S n -> 1+int_of_nat n
let bounded_int bound s = let n=int_of_string s in
  if n<0 || n>bound then invalid_arg "reference adapter size limit"; n
let e_list xs = List.fold_right (fun x xs -> E.Cons (x,xs)) xs E.Nil
let rec of_e_list = function E.Nil -> [] | E.Cons (x,xs) -> x::of_e_list xs
let csv s = if s="" then [] else String.split_on_char ',' s
let vector s =
  let xs=csv s in if List.length xs>512 then invalid_arg "reference vector limit 512";
  e_list (List.map q_of_s xs)
let json_s s = "\"" ^ String.escaped s ^ "\""
let json_q q = json_s (s_of_q q)
let json_list xs = "[" ^ String.concat "," xs ^ "]"
let json_vec xs = json_list (List.map json_q (of_e_list xs))
let json_bool = function E.True -> "true" | E.False -> "false"
let object_ entries = "{" ^ String.concat "," (List.map (fun (k,v) -> json_s k ^ ":" ^ v) entries) ^ "}"
let matrix rows cols s =
  let values=Array.of_list (List.map q_of_s (csv s)) in
  if Array.length values<>rows*cols then invalid_arg "matrix shape mismatch";
  let zero=q_of_s "0" in fun i j ->
    let i=int_of_nat i and j=int_of_nat j in
    if i>=rows || j>=cols then zero else values.(i*cols+j)
let matrix_json rows cols a =
  json_list (List.init rows (fun i -> json_list (List.init cols (fun j -> json_q (a (nat_of_int i) (nat_of_int j))))))

let handle = function
  | ["product";xs;upstream] ->
    let xs=vector xs and upstream=q_of_s upstream in
    let E.Pair(value, pb)=E.product_pullback xs upstream in
    object_ ["value",json_q value; "gradient",json_vec(E.fast_gradient xs);
      "reference",json_vec(E.reference_gradient xs); "pullback",json_vec pb]
  | ["step";xs;target;eta;fuel] ->
    let xs=vector xs and target=q_of_s target and eta=q_of_s eta in
    let fuel=nat_of_int (bounded_int 16 fuel) in
    let safe=E.safeguarded_step fuel xs target eta in
    object_ ["gradient",json_vec(E.fast_loss_gradient xs target);
      "step",json_vec(E.fast_step xs target eta);
      "reference",json_vec(E.reference_step xs target eta);
      "safe",json_vec safe; "loss_before",json_q(E.squared_loss xs target);
      "loss_after",json_q(E.squared_loss safe target)]
  | ["relu";xs;gs] ->
    let xs=of_e_list(vector xs) and gs=of_e_list(vector gs) in
    if List.length xs<>List.length gs then invalid_arg "gradient length mismatch";
    object_ ["output",json_list(List.map (fun x -> json_q(E.relu x)) xs);
      "mask",json_list(List.map (fun x -> json_bool(E.positive_mask x)) xs);
      "gradient",json_list(List.map2 (fun x g -> json_q(E.relu_pullback x g)) xs gs);
      "packed",json_list(List.map2 (fun x g -> json_q(E.packed_relu_pullback (E.positive_mask x) g)) xs gs)]
  | ["plan";m;k;n;p] ->
    let m=e_n m and k=e_n k and n=e_n n and p=e_n p in
    object_ ["contracted",json_bool(E.choose_contracted m k n p);
      "native_work",json_s(of_e_n(E.native_work m k n p));
      "contracted_work",json_s(of_e_n(E.contracted_work m k n p));
      "selected_work",json_s(of_e_n(E.selected_work m k n p));
      "native_tape",json_s(of_e_n(E.native_tape m n));
      "selected_tape",json_s(of_e_n(E.selected_tape m k n p))]
  | ["matrix";dims;xs;as_;bs;gs] ->
    let m,k,n,p = match List.map (bounded_int 12) (csv dims) with
      | [m;k;n;p] -> m,k,n,p | _ -> invalid_arg "four dimensions required" in
    let x=matrix m k xs and a=matrix k n as_ and b=matrix n p bs and g=matrix m p gs in
    let mn=nat_of_int m and kn=nat_of_int k and nn=nat_of_int n and pn=nat_of_int p in
    let y=E.planned_forward mn kn nn pn x a b in
    let E.Pair(E.Pair(dx,da),db)=E.planned_reverse mn kn nn pn x a b g in
    let E.Pair(next_a,next_b)=E.planned_parameter_updates mn kn nn pn x a b g (q_of_s "1/10") in
    object_ ["y",matrix_json m p y; "dx",matrix_json m k dx;
      "da",matrix_json k n da; "db",matrix_json n p db;
      "next_a",matrix_json k n next_a; "next_b",matrix_json n p next_b]
  | _ -> invalid_arg "unknown command or arity"

let () =
  try while true do
    let line=read_line () in
    let answer = try handle (String.split_on_char '|' line)
      with Invalid_argument message | Failure message -> object_ ["error",json_s message] in
    print_endline answer
  done with End_of_file -> ()

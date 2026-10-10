(* Hand-written boundary adapter; tested, not formally verified.
   Zarith is used only for decimal codecs. The engine retains extracted
   arbitrary-precision integers and canonical exact rationals. *)
module E = Ojasx_refined

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
let csv s = if s="" then [] else String.split_on_char ',' s
let json_s s = "\"" ^ String.escaped s ^ "\""
let json_q q = json_s(s_of_q q)
let json_list xs = "[" ^ String.concat "," xs ^ "]"
let json_bool = function E.True -> "true" | E.False -> "false"
let object_ xs = "{" ^ String.concat "," (List.map(fun(k,v)->json_s k^":"^v) xs)^"}"
let matrix m n s =
  let a=Array.of_list(List.map q_of_s(csv s)) in
  if Array.length a<>m*n then invalid_arg "matrix shape mismatch";
  let zero=q_of_s "0" in fun i j ->
    let i=int_of_nat i and j=int_of_nat j in
    if i>=m || j>=n then zero else a.(i*n+j)
let vector n s = let a=matrix n 1 s in fun i -> a i E.O
let json_vector n f = json_list(List.init n (fun i->json_q(f(nat_of_int i))))
let handle = function
  | ["compact";kind;ns;rs;inv;zs;gammas;gs] ->
    let n=bounded_int 64 ns in
    let r=q_of_s rs and inv=q_of_s inv in
    let z=vector n zs and gamma=vector n gammas and g=vector n gs in
    let q i=E.qcmult (gamma i) (g i) in
    let pull=match kind with
      | "rms"->E.compact_rms_pull | "layer"->E.compact_layer_pull
      | _->invalid_arg "unknown compact normalization kind" in
    object_ ["dx",json_vector n (pull(nat_of_int n) r inv z q)]
  | ["gather";ns;indices;xs;gs] ->
    let n=bounded_int 64 ns in
    let idx=Array.of_list(List.map(bounded_int 63)(csv indices)) in
    if Array.exists(fun i->i>=n) idx then invalid_arg "gather index out of range";
    let k=Array.length idx in
    if k>64 then invalid_arg "reference output limit 64";
    let index j=nat_of_int idx.(int_of_nat j) in
    let x=vector n xs and g=vector k gs in
    object_ ["y",json_vector k (E.gather index x);
      "dx",json_vector n (E.scatter_sum(nat_of_int k) index g)]
  | _->invalid_arg "unknown command or arity"
let () = try while true do
  let line=read_line () in
  let answer=try handle(String.split_on_char '|' line)
    with Invalid_argument message | Failure message -> object_ ["error",json_s message] in
  print_endline answer
done with End_of_file -> ()

(* Hand-written boundary adapter; tested, not formally verified.
   Zarith is used only for decimal codecs. The engine retains extracted
   arbitrary-precision integers and canonical exact rationals. *)
module E = Ojasx_foundation

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
let csv s = if s="" then [] else String.split_on_char ',' s
let json_s s = "\"" ^ String.escaped s ^ "\""
let json_q q = json_s(s_of_q q)
let json_list xs = "[" ^ String.concat "," xs ^ "]"
let json_bool = function E.True -> "true" | E.False -> "false"
let object_ xs = "{" ^ String.concat "," (List.map(fun(k,v)->json_s k^":"^v) xs)^"}"
let quad s = match List.map q_of_s (csv s) with
  | [a;b;c;d] -> {E.a00=a;a01=b;a10=c;a11=d}
  | _ -> invalid_arg "four entries required"
let json_quad a = json_list(List.map json_q [a.E.a00;a.E.a01;a.E.a10;a.E.a11])
let matrix m n s =
  let a=Array.of_list(List.map q_of_s(csv s)) in
  if Array.length a<>m*n then invalid_arg "matrix shape mismatch";
  let zero=q_of_s "0" in fun i j ->
    let i=int_of_nat i and j=int_of_nat j in
    if i>=m || j>=n then zero else a.(i*n+j)
let vector n s = let a=matrix n 1 s in fun i -> a i E.O
let json_vector n f = json_list(List.init n (fun i->json_q(f(nat_of_int i))))
let handle = function
  | [("rank7" | "winograd" as variant);a;b;g] ->
    let a=quad a and b=quad b and g=quad g in
    let mult=ref 0 and add=ref 0 in
    let algorithm = if variant="winograd" then E.winograd else E.seven in
    let counted a b = algorithm
      (fun x y -> incr add; E.qcplus x y)
      (fun x y -> incr add; E.qcminus x y)
      (fun x y -> incr mult; E.qcmult x y) a b in
    let y=counted a b in
    let da=counted g (E.trans2 b) and db=counted (E.trans2 a) g in
    object_ ["y",json_quad y;"da",json_quad da;"db",json_quad db;
      "multiplications",string_of_int !mult;"additions",string_of_int !add]
  | ["structured";ns;rs;d;u;v;x;g] ->
    let n=bounded_int 64 ns and r=bounded_int 8 rs in
    let d=vector n d and u=matrix n r u and v=matrix n r v and x=vector n x and g=vector n g in
    let nn=nat_of_int n and rn=nat_of_int r in
    object_ ["push",json_vector n (E.structured_push nn rn d u v x);
      "pull",json_vector n (E.structured_pull nn rn d u v g)]
  | ["softmax";ps;gs] ->
    let n=List.length(csv ps) in
    if n>64 then invalid_arg "reference vector limit 64";
    let p=vector n ps and g=vector n gs in
    object_ ["pull",json_vector n (E.softmax_pull (nat_of_int n) p g)]
  | [("block7" | "blockw" as variant);dims;as_;bs] ->
    let m,k,n=match List.map(bounded_int 4)(csv dims) with
      | [m;k;n] -> m,k,n | _ -> invalid_arg "three block dimensions required" in
    let a=matrix (2*m) (2*k) as_ and b=matrix (2*k) (2*n) bs in
    let blocks rows cols f =
      let at ro co i j=f (nat_of_int(ro+int_of_nat i)) (nat_of_int(co+int_of_nat j)) in
      {E.a00=at 0 0;a01=at 0 cols;a10=at rows 0;a11=at rows cols} in
    let algorithm = if variant="blockw" then E.blockw else E.block7 in
    let out=algorithm (nat_of_int k) (blocks m k a) (blocks k n b) in
    let rows=List.init (2*m) (fun i -> json_list(List.init (2*n) (fun j ->
      let block=if i<m then (if j<n then out.E.a00 else out.E.a01)
        else (if j<n then out.E.a10 else out.E.a11) in
      json_q(block(nat_of_int(i mod m))(nat_of_int(j mod n)))))) in
    object_ ["y",json_list rows]
  | ["plan7";m;a;s;b] ->
    let m=e_n m and a=e_n a and s=e_n s and b=e_n b in
    object_ ["selected",json_bool(E.choose_seven m a s b);
      "winograd_selected",json_bool(E.choose_winograd m a s b);
      "winograd",json_s(of_e_n(E.winograd_joint_work m a));
      "native",json_s(of_e_n(E.ordinary_joint_work m a));
      "rank7",json_s(of_e_n(E.seven_joint_work m a))]
  | _ -> invalid_arg "unknown command or arity"
let () = try while true do
  let line=read_line () in
  let answer=try handle(String.split_on_char '|' line)
    with Invalid_argument message | Failure message -> object_ ["error",json_s message] in
  print_endline answer
done with End_of_file -> ()

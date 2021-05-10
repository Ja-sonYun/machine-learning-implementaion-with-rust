// pub fn derivative(f: fn(f64)->f64) -> impl Fn(f64) -> f64 {
//     // let df = derivative(|x| x.powi(2));
//     // println!("{}", df(3.));
//     let h = 1e-6;
//     move |x: f64| (f(x+h) - f(x)) / h
// }
pub fn numerical_derivative(f: fn(&Vec<f64>)->f64, x: &mut Vec<f64>) -> Vec<f64> {
    // let pf = |v: &Vec<f64>|->f64 { ( v[3]*v[0] ) + ( v[0]*v[1]*v[3] ) + ( 3.*v[2] ) + ( v[3]*v[1].powi(2) ) };
    // let mut vec: Vec<f64> = vec![2., 3., 1., 4.];
    // println!("{:?}", numerical_derivative(pf, &mut vec));
    let delta_x = 1e-4;
    let mut grad = vec![0.; x.len()];

    for i in 0..x.len() as usize {
        let tmp_val = x[i];
        x[i] = tmp_val + delta_x;
        let fx1 = f(x);

        x[i] = tmp_val - delta_x;
        let fx2 = f(x);

        grad[i] = (fx1 - fx2) / (2. * delta_x);
        x[i] = tmp_val;
    }
    grad
}

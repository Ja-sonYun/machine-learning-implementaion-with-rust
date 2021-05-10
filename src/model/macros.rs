#[macro_export]
macro_rules! add_layer {
    ($model: ident, $layer: expr, $in: expr, $out: expr, $la: expr, $act: expr, $name: expr) => {
        $model.add_layer($layer, $in, $out, $la, $act, Some($name))
    };
}

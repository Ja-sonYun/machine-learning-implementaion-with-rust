#[macro_escape]

#[macro_export]
macro_rules! forfor {
    ($y: expr, $yn: ident, $x: expr, $xn: ident, $bk: block) => {
        for $yn in 0..$y {
            for $xn in 0..$x {
                $bk
            }
        }
    }
}

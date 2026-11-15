/// The build script translates templates to Rust format literals. The compiler
/// checks that each call supplies exactly the template's named parameters.
macro_rules! wgsl {
    (to = $output:expr, $path:literal, $($name:ident = $value:expr),+ $(,)?) => {
        write!(
            $output,
            include!(concat!(env!("OUT_DIR"), "/wgsl/", $path, ".rs")),
            $($name = $value),+
        ).unwrap()
    };
    ($path:literal, $($name:ident = $value:expr),+ $(,)?) => {
        format!(
            include!(concat!(env!("OUT_DIR"), "/wgsl/", $path, ".rs")),
            $($name = $value),+
        )
    };
}

pub(crate) use wgsl;

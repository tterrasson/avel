use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use avel_compiler::{CompileOptions, Features, Graph, Limits, compile};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "avel", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a graph into an execution plan.
    Compile {
        /// Path to a `model.graph.json` file.
        graph: PathBuf,
        /// Concrete input shape, as `name=1,3,224,224` or `name=` for a scalar. Repeatable.
        #[arg(long = "input", value_parser = parse_input)]
        inputs: Vec<(String, Vec<u64>)>,
        /// Where to write the plan. Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Store activations in f16, for devices with `shader-f16`.
        #[arg(long)]
        f16: bool,
        /// Use f16 GEMM products with f32 accumulation.
        #[arg(long)]
        f16_compute: bool,
        /// Snap symbolic input dimensions up to a multiple of this, so
        /// variable-length sequences share plans within a bucket.
        #[arg(long)]
        bucket: Option<u64>,
    },
}

fn parse_input(value: &str) -> Result<(String, Vec<u64>), String> {
    let (name, shape) = value
        .split_once('=')
        .ok_or_else(|| format!("expected name=shape, got `{value}`"))?;
    if name.is_empty() {
        return Err("input name must not be empty".into());
    }
    if shape.is_empty() {
        return Ok((name.to_owned(), Vec::new()));
    }
    let shape = shape
        .split(',')
        .map(|dim| {
            dim.parse()
                .map_err(|_| format!("invalid dimension `{dim}`"))
        })
        .collect::<Result<_, _>>()?;
    Ok((name.to_owned(), shape))
}

fn main() -> Result<(), Box<dyn Error>> {
    let Command::Compile {
        graph,
        inputs,
        output,
        f16,
        f16_compute,
        bucket,
    } = Cli::parse().command;

    let graph: Graph = serde_json::from_str(&fs::read_to_string(graph)?)?;
    let mut input_shapes = BTreeMap::new();
    for (name, shape) in inputs {
        use std::collections::btree_map::Entry;
        match input_shapes.entry(name) {
            Entry::Vacant(entry) => {
                entry.insert(shape);
            }
            Entry::Occupied(entry) => {
                return Err(
                    format!("input shape for `{}` was given more than once", entry.key()).into(),
                );
            }
        }
    }
    let options = CompileOptions {
        input_shapes,
        limits: Limits::default(),
        features: Features {
            shader_f16: f16 || f16_compute,
            ..Features::default()
        },
        f16,
        f16_compute,
        bucket,
        media: BTreeMap::new(),
        custom_kernels: BTreeMap::new(),
        gpu_class: Default::default(),
        tiles: BTreeMap::new(),
    };
    let plan = compile(&graph, &options)?;
    match output {
        Some(path) => {
            let mut file = BufWriter::new(fs::File::create(path)?);
            serde_json::to_writer_pretty(&mut file, &plan)?;
            file.flush()?;
        }
        None => {
            let mut stdout = BufWriter::new(io::stdout().lock());
            serde_json::to_writer_pretty(&mut stdout, &plan)?;
            writeln!(stdout)?;
            stdout.flush()?;
        }
    }
    Ok(())
}

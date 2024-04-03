use std::any::TypeId;
use std::fs::File;
use std::io::BufReader;

use bft_interp::{BrainfuckVM, VMBuilder};

use criterion::{criterion_group, criterion_main, Criterion};

fn my_benchmark_function(c: &mut Criterion) {
    c.bench_function("my_benchmark", |b| {
        b.iter(|| {
            let file =
                File::open("/home/sam/git/rust-homework-3/bench.bf").expect("Failed to open file");
            let program_file = BufReader::new(file);

            let mut vm: BrainfuckVM<u8> = VMBuilder::<BufReader<File>, std::io::Stdout>::new()
                .set_program_file(program_file)
                .set_allow_growth(true)
                .set_cell_kind(TypeId::of::<u8>())
                .build()
                .expect("Failed to build VM"); // Here's the change

            vm.interpret().expect("Interpretation failed"); // Assuming `interpret` might also return a Result
        });
    });
}

criterion_group!(benches, my_benchmark_function);
criterion_main!(benches);

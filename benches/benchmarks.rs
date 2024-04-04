use std::io::Cursor;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use bft_interp::vm::BrainfuckVM;
use bft_interp::vm_builder::VMBuilder;
use bft_test_utils::NullWriter;
use bft_types::bf_program::Program;
use criterion::black_box;
use criterion::{criterion_group, criterion_main, Criterion};

fn interpreter_throughput(c: &mut Criterion) {
    // Just do a hello world program
    let program_string = "++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<+++++++++++++++.>.+++.------.--------.>+.>.";

    c.bench_function("hello_world", |b| {
        b.iter(|| {
            let mut vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
                .set_program_reader(Cursor::new(program_string))
                .set_cell_count(NonZeroUsize::new(30000))
                .build()
                .unwrap();
            vm.interpret().unwrap();
        });
    });
}

fn fixed_memory(c: &mut Criterion) {
    // Fixed tape size, move head right until the end of the tape (29,999)
    // Do a + at the end just to actually using the final cell
    let program_string = ">".repeat(30000 - 1) + "+";
    c.bench_function("fixed_memory", |b| {
        b.iter(|| {
            let mut vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
                .set_program_reader(Cursor::new(black_box(&program_string)))
                .set_allow_growth(false) // Note: `set_allow_growth(false)` for fixed memory
                .build()
                .unwrap();
            vm.interpret().unwrap();
        });
    });
}

fn memory_growth(c: &mut Criterion) {
    // Allow growth, move head 3 times more than the fixed memory test
    // Do a + at the end just to actually using the final cell
    let program_string = ">".repeat(120000 - 3) + "+";
    c.bench_function("memory_growth", |b| {
        b.iter(|| {
            let mut vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
                .set_program_reader(Cursor::new(black_box(&program_string)))
                .set_allow_growth(true)
                .build()
                .unwrap();
            vm.interpret().unwrap();
        });
    });
}

fn nested_loops(c: &mut Criterion) {
    // Lots of nested loops
    let program_string =
        "[[[[[[[[[[-]>]>>>>]<<<<<<]>>>>>>]<<<<<<<<<]>>>>>>>>>]<<<<<<<<<<]>>>>>>>>>>>]<<<<<<<<<<<]";
    c.bench_function("nested_loops", |b| {
        b.iter(|| {
            let mut vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
                .set_program_reader(Cursor::new(black_box(&program_string)))
                .set_allow_growth(true)
                .build()
                .unwrap();
            vm.interpret().unwrap();
        });
    });
}

fn long_program(c: &mut Criterion) {
    // Fibonacci sequence
    // Using NullWriter because this program spits loads of stuff out
    c.bench_function("long_program", |b| {
        b.iter(|| {
            let mut vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, NullWriter>::new()
                .set_program_file(PathBuf::from("benches/fib.bf"))
                .set_allow_growth(true)
                .set_output(NullWriter)
                .build()
                .unwrap();
            vm.interpret().unwrap();
        });
    });
}

criterion_group!(memory_benchmarks, fixed_memory, memory_growth);
criterion_group!(complexity_benchmarks, nested_loops, long_program);
criterion_group!(throughput_benchmarks, interpreter_throughput);
criterion_group!(
    all_benchmarks,
    fixed_memory,
    memory_growth,
    nested_loops,
    long_program,
    interpreter_throughput
);

criterion_main!(all_benchmarks);

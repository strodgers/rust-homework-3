use bft_interp::builder::VMBuilder;
use bft_interp::core::BrainfuckVM;
use criterion::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use std::io::{self, Cursor, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;

pub struct NullWriter;

impl Write for NullWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Pretend everything's okay and we wrote the whole buffer.
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Nothing to flush, so just say it worked.
        Ok(())
    }
}

fn interpreter_throughput(c: &mut Criterion) {
    // Just do a hello world program
    let program_string = "++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<+++++++++++++++.>.+++.------.--------.>+.>.";

    c.bench_function("hello_world", |b| {
        b.iter(|| {
            let mut vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, NullWriter>::new()
                .set_program_reader(Cursor::new(program_string))
                .set_cell_count(NonZeroUsize::new(30000))
                .set_output(NullWriter)
                .set_optimization(false) // Small program
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
                .set_optimization(false) // Fixed memory, no point
                .build()
                .expect("Failed to build VM");
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
                .set_optimization(true)
                .build()
                .expect("Failed to build VM");
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
                .set_optimization(true)
                .build()
                .expect("Failed to build VM");
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
                .set_optimization(true)
                .set_buffer_output(true)
                .build()
                .expect("Failed to build VM");
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

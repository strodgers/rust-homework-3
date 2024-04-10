# Notes

## Before anything

When I started, there were many clippy warnings visible immediately.

Some were unnecessary borrows, some were string extensions, there was
also an unnecessary lifetime and an iterator changeover.

To reduce confusion I cleared all the warnings up and made a commit: `b3c8e48`

## Test outcomes

- `--help` output looks sane (if huge)
- Output is bad (outputs extraneous line of text)
- Basic newline wrapped output is missing (hw/hw2)
- Extensible tape is good (-c 1 -e hw.bf)
- Flushing is good (primes)
- Performance is good (primes)
- General behaviour good (game)

Testing "failed" but not critically

## Documentation review

`cargo doc --workspace --open --no-deps`

- No warnings while building docs, which is nice.
- Lots of modules; perhaps hiding those and reexporting contents might be better?
- Actual docs are very sparse, particularly at function level, sometimes at the
  crate/module level but also sometimes at type level.

Documentation is therefore "fail".

## Code skim

- Complexity is through the roof as discussed before. I understand why.
- Generally splitting stuff into modules for code arrangement seems arbitrary or pythonic
  rather than rusty.
- use of foo/mod.rs rather than foo.rs was also a bit clunky, the codebase felt "old"
  and not in a "well lived in" way.
- Generally speaking there was a _lot_ of dense and hard to read code.
- Interpreter does not handle EOF on input cleanly
  - Propose future work, read <https://esolangs.org/wiki/Brainfuck#EOF_2> and decide

I _want_ to fail it, but it's not awful. So I'll grudgingly pass on the basis that you
go ahead and fix the output (extraneous) and newline stuff which is absent.
I'd like to see _some_ docs but at this point I won't force the issue.

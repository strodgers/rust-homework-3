use std::io::{self, Read, Seek, SeekFrom, Write};
use tempfile::NamedTempFile;

const TEST_FILE_CONTENT: &str = "+[-[<<[+[--->]-[<<<]]]>>>-]";
// writelin! does add an extra newline character at the end, but will ignore
// that since really only interested in the number of BF instructions
pub const TEST_FILE_NUM_INSTRUCTIONS: usize = TEST_FILE_CONTENT.len();
pub struct TestFile {
    file: NamedTempFile,
}

impl TestFile {
    pub fn new() -> io::Result<Self> {
        let mut file = NamedTempFile::new()?;
        // Don't do any input/output for this test
        writeln!(file, "{}", TEST_FILE_CONTENT)?;

        // Seek to the start of the file after writing, also get the length
        file.seek(SeekFrom::Start(0))?;
        Ok(TestFile { file })
    }
}

impl Read for TestFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Need to borrow it mutably to perform reads
        self.file.as_file_mut().read(buf)
    }
}

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

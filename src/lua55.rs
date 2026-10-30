//! Lua 5.5 chunks (ldump.c).  Compared with 5.4: varints mark every byte but
//! the last, integer constants are zigzag varints, every string is written
//! once and later referenced by index, the code array (and absolute line
//! info) is aligned to 4 bytes from the start of the dump, and a function's
//! source name comes after its prototypes.  Parsed by hand: the string table
//! and the alignment need the reader's state and absolute position.

use super::*;

/// Lua 5.5 `PF_VAHID | PF_VATAB`: the function is vararg.
const PF_VARARG: u8 = 1 | 2;

type R<T> = Result<T, String>;

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Strings in the order they were saved (references are 1-based).
    strings: Vec<Vec<u8>>,
}

impl<'a> Reader<'a> {
    fn err<T>(&self, what: &str) -> R<T> {
        Err(format!("lua 5.5 chunk: {what} at byte {}", self.pos))
    }

    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        match self.pos.checked_add(n) {
            Some(end) if end <= self.data.len() => {
                let s = &self.data[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            _ => self.err("unexpected end of data"),
        }
    }

    fn byte(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }

    /// MSB-first groups of 7 bits; the high bit marks "more bytes follow".
    fn varint(&mut self) -> R<u64> {
        let mut x: u64 = 0;
        loop {
            let b = self.byte()?;
            if x > u64::MAX >> 7 {
                return self.err("integer overflow");
            }
            x = (x << 7) | (b & 0x7f) as u64;
            if b & 0x80 == 0 {
                return Ok(x);
            }
        }
    }

    fn count(&mut self) -> R<usize> {
        let n = self.varint()?;
        if n > (self.data.len() - self.pos) as u64 {
            return self.err("count larger than the rest of the chunk");
        }
        Ok(n as usize)
    }

    /// 0 => 0, -1 => 1, 1 => 2, -2 => 3, ...
    fn integer(&mut self) -> R<i64> {
        let c = self.varint()?;
        Ok(if c & 1 == 0 { (c >> 1) as i64 } else { !(c >> 1) as i64 })
    }

    fn align(&mut self, a: usize) -> R<()> {
        let pad = (a - self.pos % a) % a;
        self.take(pad).map(|_| ())
    }

    /// `None` for a NULL string.
    fn string(&mut self) -> R<Option<Vec<u8>>> {
        let size = self.varint()?;
        if size == 0 {
            let idx = self.varint()? as usize;
            if idx == 0 {
                return Ok(None);
            }
            return match self.strings.get(idx - 1) {
                Some(s) => Ok(Some(s.clone())),
                None => self.err("reference to an unknown saved string"),
            };
        }
        // `size` counts the terminating '\0', which is written too.
        let bytes = self.take(size as usize)?;
        let s = bytes[..bytes.len() - 1].to_vec();
        self.strings.push(s.clone());
        Ok(Some(s))
    }

    fn constant(&mut self) -> R<LuaConstant> {
        Ok(match self.byte()? {
            0x00 => LuaConstant::Null,
            0x01 => LuaConstant::Bool(false),
            0x11 => LuaConstant::Bool(true),
            0x03 => LuaConstant::Number(LuaNumber::Integer(self.integer()?)),
            0x13 => {
                let b = self.take(8)?;
                LuaConstant::Number(LuaNumber::Float(f64::from_le_bytes(b.try_into().unwrap())))
            }
            0x04 | 0x14 => match self.string()? {
                Some(s) => LuaConstant::from(s),
                None => return self.err("NULL string constant"),
            },
            _ => return self.err("unknown constant tag"),
        })
    }

    fn function(&mut self, depth: usize) -> R<LuaChunk> {
        if depth > 200 {
            return self.err("functions nested too deeply");
        }
        let line_defined = self.varint()?;
        let last_line_defined = self.varint()?;
        let num_params = self.byte()?;
        let flags = self.byte()?;
        let max_stack = self.byte()?;

        let ncode = self.count()?;
        self.align(4)?;
        let code = self.take(ncode.checked_mul(4).ok_or_else(|| "code size".to_string())?)?;
        let instructions = code.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();

        let nk = self.count()?;
        let constants = (0..nk).map(|_| self.constant()).collect::<R<Vec<_>>>()?;

        let nup = self.count()?;
        let mut upvalue_infos = Vec::with_capacity(nup);
        for _ in 0..nup {
            let b = self.take(3)?;
            upvalue_infos.push(UpVal { on_stack: b[0] != 0, id: b[1], kind: b[2] });
        }

        let np = self.count()?;
        let prototypes = (0..np).map(|_| self.function(depth + 1)).collect::<R<Vec<_>>>()?;

        let name = self.string()?.unwrap_or_default();

        // Debug information.
        let nline = self.count()?;
        self.take(nline)?;
        let nabs = self.count()?;
        let mut source_lines = Vec::with_capacity(nabs);
        if nabs > 0 {
            self.align(4)?;
            for _ in 0..nabs {
                let b = self.take(8)?;
                let pc = u32::from_le_bytes(b[0..4].try_into().unwrap());
                let line = u32::from_le_bytes(b[4..8].try_into().unwrap());
                source_lines.push((pc, line));
            }
        }
        let nloc = self.count()?;
        let mut locals = Vec::with_capacity(nloc);
        for _ in 0..nloc {
            let name = self.string()?.unwrap_or_default();
            let start_pc = self.varint()?;
            let end_pc = self.varint()?;
            locals.push(LuaLocal { name: String::from_utf8_lossy(&name).into(), start_pc, end_pc, reg: 0 });
        }
        let nupn = self.count()?;
        let upvalue_names = (0..nupn).map(|_| self.string().map(Option::unwrap_or_default)).collect::<R<Vec<_>>>()?;

        Ok(LuaChunk {
            name,
            line_defined,
            last_line_defined,
            num_upvalues: upvalue_infos.len() as u8,
            num_params,
            max_stack,
            flags,
            is_vararg: if flags & PF_VARARG != 0 { Some(LuaVarArgInfo::new()) } else { None },
            instructions,
            constants,
            num_constants: vec![],
            prototypes,
            source_lines,
            locals,
            upvalue_infos,
            upvalue_names,
        })
    }
}

/// Is this a Lua 5.5 chunk?
pub fn is_lua55(input: &[u8]) -> bool {
    input.starts_with(b"\x1bLua\x55")
}

pub fn parse(input: &[u8]) -> Result<LuaBytecode, String> {
    let mut r = Reader { data: input, pos: 5, strings: Vec::new() };
    let format_version = r.byte()?;
    if r.take(6)? != b"\x19\x93\r\n\x1a\n" {
        return r.err("corrupted header");
    }
    // Sizes (each followed by a sample value) of int, Instruction,
    // lua_Integer and lua_Number.
    let mut sizes = [0u8; 4];
    for s in sizes.iter_mut() {
        *s = r.byte()?;
        r.take(*s as usize)?;
    }
    let [int_size, instruction_size, integer_size, number_size] = sizes;
    if instruction_size != 4 || integer_size != 8 || number_size != 8 {
        return r.err("unsupported number or instruction size");
    }
    let _num_upvalues = r.byte()?;
    let main_chunk = r.function(0)?;
    let header = LuaHeader {
        lua_version: LUA55.0,
        format_version,
        big_endian: false,
        int_size,
        size_t_size: 8,
        instruction_size,
        number_size,
        number_integral: false,
        ..Default::default()
    };
    Ok(LuaBytecode { header, main_chunk })
}

//! LuaJIT bytecode dumps (src/lj_bcread.c, lj_bcdump.h), dump versions 1
//! (LuaJIT 2.0) and 2 (LuaJIT 2.1).
//!
//! A dump is a header followed by prototypes in post-order: every child
//! comes before its parent and is popped off a stack by the parent's
//! `BCDUMP_KGC_CHILD` constants; the last prototype is the main function.
//!
//! In the resulting [`LuaChunk`]:
//! - `instructions` are the dumped words (the `FUNCF`/`FUNCV` header
//!   instruction at pc 0 is not stored, so word `i` is pc `i + 1`);
//! - `constants` are the GC constants in the order the bytecode indexes them
//!   (operand `D` names `constants[D]`), children as [`LuaConstant::Proto`];
//! - `num_constants` are the number constants (`KNUM` etc.);
//! - `upvalue_infos` keep LuaJIT's upvalue references; `locals` and
//!   `upvalue_names` come from the debug info, local pcs shifted to word
//!   indices.

use super::*;

/* Header flags of bytecode */
pub const FLAG_IS_BIG_ENDIAN: u8 = 0x01;
pub const FLAG_IS_STRIPPED: u8 = 0x02;
pub const FLAG_HAS_FFI: u8 = 0x04;
/// LuaJIT 2.1 with two-slot frames (64-bit GC): call arguments start at
/// `base + 2`.
pub const FLAG_F_FR2: u8 = 0x08;
/// LuaJIT 2.1 bitwise operator instructions (`BAND` ...) may occur.
pub const FLAG_F_BITOP: u8 = 0x10;

/* Flags for prototype. */
pub const PROTO_CHILD: u8 = 0x01; /* Has child prototypes. */
pub const PROTO_VARARG: u8 = 0x02; /* Vararg function. */
pub const PROTO_FFI: u8 = 0x04; /* Uses BC_KCDATA for FFI datatypes. */
pub const PROTO_NOJIT: u8 = 0x08; /* JIT disabled for this function. */
pub const PROTO_ILOOP: u8 = 0x10; /* Patched bytecode with ILOOP etc. */

/// Upvalue reference: from the parent's stack slot (else its upvalue).
pub const PROTO_UV_LOCAL: u16 = 0x8000;
/// Upvalue reference: the variable is never assigned after capture.
pub const PROTO_UV_IMMUTABLE: u16 = 0x4000;

pub const BCDUMP_KGC_CHILD: u32 = 0;
pub const BCDUMP_KGC_TAB: u32 = 1;
pub const BCDUMP_KGC_I64: u32 = 2;
pub const BCDUMP_KGC_U64: u32 = 3;
pub const BCDUMP_KGC_COMPLEX: u32 = 4;
pub const BCDUMP_KGC_STR: u32 = 5;

pub const BCDUMP_KTAB_NIL: u32 = 0;
pub const BCDUMP_KTAB_FALSE: u32 = 1;
pub const BCDUMP_KTAB_TRUE: u32 = 2;
pub const BCDUMP_KTAB_INT: u32 = 3;
pub const BCDUMP_KTAB_NUM: u32 = 4;
pub const BCDUMP_KTAB_STR: u32 = 5;

/// Names of LuaJIT's internal variables (VARNAMEDEF in lj_parse.c), by
/// their varinfo code 1..=6.
const VARNAMES: [&str; 6] = [
    "(for index)",
    "(for limit)",
    "(for step)",
    "(for generator)",
    "(for state)",
    "(for control)",
];

type R<T> = Result<T, String>;

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    big_endian: bool,
}

impl<'a> Reader<'a> {
    fn err<T>(&self, what: &str) -> R<T> {
        Err(format!("luajit bytecode: {what} at byte {}", self.pos))
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

    fn u8(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> R<u16> {
        let b: [u8; 2] = self.take(2)?.try_into().unwrap();
        Ok(if self.big_endian { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
    }

    fn u32(&mut self) -> R<u32> {
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.big_endian { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) })
    }

    /// ULEB128, at most 32 bits (lj_buf_ruleb128).
    fn uleb(&mut self) -> R<u32> {
        let mut v: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            if shift < 35 {
                v |= ((b & 0x7f) as u64) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                return Ok(v as u32);
            }
            if shift > 63 {
                return self.err("uleb128 too long");
            }
        }
    }

    /// The 33-bit ULEB128 of number constants: the low bit of the first
    /// byte is the "is a double" flag (bcread_uleb128_33).
    fn uleb33(&mut self) -> R<u32> {
        let first = self.u8()?;
        let mut v = (first >> 1) as u32;
        if v >= 0x40 {
            v &= 0x3f;
            let mut sh = 6;
            loop {
                let b = self.u8()?;
                if sh < 32 {
                    v |= ((b & 0x7f) as u32) << sh;
                }
                sh += 7;
                if b < 0x80 {
                    break;
                }
                if sh > 63 {
                    return self.err("uleb128 too long");
                }
            }
        }
        Ok(v)
    }

    fn count(&mut self) -> R<usize> {
        let n = self.uleb()? as usize;
        if n > self.data.len() - self.pos {
            return self.err("count larger than the rest of the dump");
        }
        Ok(n)
    }

    /// Two ULEB128 halves of a 64-bit value (low word first).
    fn u64_halves(&mut self) -> R<u64> {
        let lo = self.uleb()? as u64;
        let hi = self.uleb()? as u64;
        Ok(lo | (hi << 32))
    }

    fn ktabk(&mut self) -> R<LuaConstant> {
        let tp = self.uleb()?;
        Ok(match tp {
            BCDUMP_KTAB_NIL => LuaConstant::Null,
            BCDUMP_KTAB_FALSE => LuaConstant::Bool(false),
            BCDUMP_KTAB_TRUE => LuaConstant::Bool(true),
            BCDUMP_KTAB_INT => LuaConstant::Number(LuaNumber::Integer(self.uleb()? as i32 as i64)),
            BCDUMP_KTAB_NUM => LuaConstant::Number(LuaNumber::Float(f64::from_bits(self.u64_halves()?))),
            _ => {
                let len = (tp - BCDUMP_KTAB_STR) as usize;
                LuaConstant::from(self.take(len)?.to_vec())
            }
        })
    }

    fn ktab(&mut self) -> R<LuaConstant> {
        let narray = self.count()?;
        let nhash = self.count()?;
        let mut array = Vec::with_capacity(narray);
        for _ in 0..narray {
            array.push(self.ktabk()?);
        }
        let mut hash = Vec::with_capacity(nhash);
        for _ in 0..nhash {
            let k = self.ktabk()?;
            let v = self.ktabk()?;
            hash.push((k, v));
        }
        Ok(LuaConstant::Table(Box::new(ConstTable { array, hash })))
    }

    /// NUL-terminated string.
    fn cstr(&mut self) -> R<&'a [u8]> {
        let rest = &self.data[self.pos..];
        match rest.iter().position(|&b| b == 0) {
            Some(n) => {
                let s = &rest[..n];
                self.pos += n + 1;
                Ok(s)
            }
            None => self.err("unterminated name"),
        }
    }

    /// One prototype (the length prefix already read); `stack` holds the
    /// finished prototypes not yet claimed by a parent.
    fn proto(&mut self, flags: u8, stack: &mut Vec<LuaChunk>) -> R<LuaChunk> {
        let pflags = self.u8()?;
        let numparams = self.u8()?;
        let framesize = self.u8()?;
        let sizeuv = self.u8()? as usize;
        let sizekgc = self.count()?;
        let sizekn = self.count()?;
        let sizebc = self.count()?; // without the header instruction
        let (mut sizedbg, mut firstline, mut numline) = (0, 0, 0);
        if flags & FLAG_IS_STRIPPED == 0 {
            sizedbg = self.count()?;
            if sizedbg > 0 {
                firstline = self.uleb()?;
                numline = self.uleb()?;
            }
        }

        let mut instructions = Vec::with_capacity(sizebc);
        for _ in 0..sizebc {
            instructions.push(self.u32()?);
        }
        let mut upvalue_infos = Vec::with_capacity(sizeuv);
        for _ in 0..sizeuv {
            let v = self.u16()?;
            upvalue_infos.push(UpVal {
                on_stack: v & PROTO_UV_LOCAL != 0,
                id: (v & 0xff) as u8,
                kind: (v & PROTO_UV_IMMUTABLE != 0) as u8,
            });
        }

        // GC constants are stored lowest address first; operand D names
        // the D-th one counting back from the end.
        let mut kgc = Vec::with_capacity(sizekgc);
        let mut prototypes = Vec::new();
        for _ in 0..sizekgc {
            let tp = self.uleb()?;
            kgc.push(match tp {
                BCDUMP_KGC_CHILD => match stack.pop() {
                    Some(child) => {
                        prototypes.push(child);
                        LuaConstant::Proto(prototypes.len() - 1)
                    }
                    None => return self.err("child prototype missing"),
                },
                BCDUMP_KGC_TAB => self.ktab()?,
                BCDUMP_KGC_I64 => LuaConstant::Int64(self.u64_halves()? as i64),
                BCDUMP_KGC_U64 => LuaConstant::UInt64(self.u64_halves()?),
                BCDUMP_KGC_COMPLEX => {
                    let re = f64::from_bits(self.u64_halves()?);
                    let im = f64::from_bits(self.u64_halves()?);
                    LuaConstant::Complex(re, im)
                }
                _ => {
                    let len = (tp - BCDUMP_KGC_STR) as usize;
                    LuaConstant::from(self.take(len)?.to_vec())
                }
            });
        }
        kgc.reverse();
        // Children were popped last-dumped first; number them in source
        // order instead.
        let np = prototypes.len();
        prototypes.reverse();
        for k in kgc.iter_mut() {
            if let LuaConstant::Proto(i) = k {
                *i = np - 1 - *i;
            }
        }

        let mut num_constants = Vec::with_capacity(sizekn);
        for _ in 0..sizekn {
            let isnum = self.data.get(self.pos).map_or(false, |b| b & 1 != 0);
            let lo = self.uleb33()?;
            num_constants.push(if isnum {
                let hi = self.uleb()?;
                LuaNumber::Float(f64::from_bits(lo as u64 | ((hi as u64) << 32)))
            } else {
                LuaNumber::Integer(lo as i32 as i64)
            });
        }

        let mut upvalue_names = Vec::new();
        let mut locals = Vec::new();
        if sizedbg > 0 {
            let end = self.pos + sizedbg;
            let shift = if numline < 256 { 0 } else if numline < 65536 { 1 } else { 2 };
            self.take(sizebc << shift)?; // line info
            for _ in 0..sizeuv {
                upvalue_names.push(self.cstr()?.to_vec());
            }
            let mut lastpc: u64 = 0;
            while self.pos < end {
                let code = self.data[self.pos];
                let name = if code == 0 {
                    break;
                } else if (code as usize) <= VARNAMES.len() {
                    self.pos += 1;
                    VARNAMES[code as usize - 1].to_string()
                } else {
                    String::from_utf8_lossy(self.cstr()?).into_owned()
                };
                let start = lastpc + self.uleb()? as u64;
                let end_pc = start + self.uleb()? as u64;
                lastpc = start;
                // pcs count the header instruction; word i is pc i + 1.
                locals.push(LuaLocal {
                    name,
                    start_pc: start.saturating_sub(1),
                    end_pc: end_pc.saturating_sub(1),
                    reg: 0,
                });
            }
            if end > self.data.len() {
                return self.err("debug info past the end");
            }
            self.pos = end;
        }

        Ok(LuaChunk {
            name: vec![],
            line_defined: firstline as u64,
            last_line_defined: firstline as u64 + numline as u64,
            num_upvalues: sizeuv as u8,
            num_params: numparams,
            max_stack: framesize,
            flags: pflags,
            is_vararg: if pflags & PROTO_VARARG != 0 { Some(LuaVarArgInfo::new()) } else { None },
            instructions,
            constants: kgc,
            num_constants,
            prototypes,
            source_lines: vec![],
            locals,
            upvalue_infos,
            upvalue_names,
        })
    }
}

pub fn is_luajit(input: &[u8]) -> bool {
    input.starts_with(b"\x1bLJ")
}

pub fn parse(input: &[u8]) -> Result<LuaBytecode, String> {
    let mut r = Reader { data: input, pos: 3, big_endian: false };
    let version = r.u8()?;
    let lua_version = match version {
        1 => LUAJ1,
        2 => LUAJ2,
        v => return Err(format!("unsupported LuaJIT bytecode version {v}")),
    };
    let flags = r.uleb()?;
    if flags & !0x1f != 0 {
        return Err(format!("unknown LuaJIT bytecode flags {flags:#x}"));
    }
    let flags = flags as u8;
    r.big_endian = flags & FLAG_IS_BIG_ENDIAN != 0;
    let mut name = Vec::new();
    if flags & FLAG_IS_STRIPPED == 0 {
        let len = r.count()?;
        name = r.take(len)?.to_vec();
    }
    let mut stack: Vec<LuaChunk> = Vec::new();
    while r.pos < input.len() {
        let len = r.count()?;
        if len == 0 {
            break;
        }
        let start = r.pos;
        let proto = r.proto(flags, &mut stack)?;
        if r.pos != start + len {
            return Err(format!("luajit bytecode: prototype at byte {start} has the wrong length"));
        }
        stack.push(proto);
    }
    let mut main_chunk = match (stack.pop(), stack.is_empty()) {
        (Some(main), true) => main,
        _ => return Err("luajit bytecode: unbalanced prototypes".into()),
    };
    main_chunk.name = name;
    Ok(LuaBytecode {
        header: LuaHeader {
            lua_version: lua_version.0,
            format_version: version,
            big_endian: r.big_endian,
            int_size: 4,
            size_t_size: 4,
            instruction_size: 4,
            number_size: 8,
            number_integral: false,
            lj_flags: flags,
        },
        main_chunk,
    })
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ProtoFlags: u8 {
        const HAS_CHILD = 0b00000001;
        const IS_VARIADIC = 0b00000010;
        const HAS_FFI = 0b00000100;
        const JIT_DISABLED = 0b00001000;
        const HAS_ILOOP = 0b00010000;
        const BITOP = 0b10000000;
    }
}

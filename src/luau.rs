//! Luau bytecode (VM/src/lvmload.cpp, `loadsafe`), bytecode versions 3..14.
//!
//! One string table and one flat proto table per chunk; protos refer to
//! their children (and closure constants to protos) by index into that
//! table.  The result is the usual tree of [`LuaChunk`]s: child protos in
//! `prototypes`, and closure constants rewritten to the child's position
//! there.  Opcodes are left as stored (Roblox clients scramble them; see
//! the decompiler's front end).

use super::*;

pub const LBC_CONSTANT_NIL: u8 = 0;
pub const LBC_CONSTANT_BOOLEAN: u8 = 1;
pub const LBC_CONSTANT_NUMBER: u8 = 2;
pub const LBC_CONSTANT_STRING: u8 = 3;
pub const LBC_CONSTANT_IMPORT: u8 = 4;
pub const LBC_CONSTANT_TABLE: u8 = 5;
pub const LBC_CONSTANT_CLOSURE: u8 = 6;
pub const LBC_CONSTANT_VECTOR: u8 = 7;
pub const LBC_CONSTANT_TABLE_WITH_CONSTANTS: u8 = 8;
pub const LBC_CONSTANT_INTEGER: u8 = 9;
pub const LBC_CONSTANT_CLASS_SHAPE: u8 = 10;
pub const LBC_CONSTANT_VECTORD: u8 = 11;

/// `LPF_INLINABLE`: the proto carries a cost (bytecode version 12+).
const LPF_INLINABLE: u8 = 1 << 3;

type R<T> = Result<T, String>;

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn err<T>(&self, what: &str) -> R<T> {
        Err(format!("luau bytecode: {what} at byte {}", self.pos))
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

    fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> R<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> R<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// LEB128: least significant group first, high bit = more bytes follow.
    fn varint(&mut self) -> R<u64> {
        let mut x: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            if shift >= 64 {
                return self.err("varint too long");
            }
            x |= ((b & 0x7f) as u64) << shift;
            shift += 7;
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

    /// A reference into the string table (0 = none).
    fn string(&mut self, strings: &[Rc<ByteBuf>]) -> R<Option<Rc<ByteBuf>>> {
        let i = self.varint()? as usize;
        if i == 0 {
            return Ok(None);
        }
        match strings.get(i - 1) {
            Some(s) => Ok(Some(s.clone())),
            None => self.err("string index out of range"),
        }
    }
}

/// Does this look like Luau bytecode (or a Luau compile error)?
pub fn is_luau(input: &[u8]) -> bool {
    match input.first() {
        Some(&v) => v == 0 || (3..=14).contains(&v) || v == 100,
        None => false,
    }
}

struct RawProto {
    chunk: LuaChunk,
    /// Global proto ids of the children, in order.
    children: Vec<usize>,
    /// Constant slots holding closures, with their global proto id.
    closure_consts: Vec<(usize, usize)>,
}

pub fn parse(input: &[u8]) -> Result<LuaBytecode, String> {
    let mut r = Reader { data: input, pos: 0 };
    let version = r.u8()?;
    if version == 0 {
        // The rest is the compiler's error message.
        return Err(format!("luau compile error: {}", String::from_utf8_lossy(&input[1..])));
    }
    let types_version = if version >= 4 { r.u8()? } else { 0 };

    let nstr = r.count()?;
    let mut strings = Vec::with_capacity(nstr);
    for _ in 0..nstr {
        let n = r.count()?;
        strings.push(Rc::new(ByteBuf::from(r.take(n)?.to_vec())));
    }

    // Userdata type names (types version 3): index byte, name, ..., 0.
    if types_version == 3 {
        while r.u8()? != 0 {
            r.string(&strings)?;
        }
    }

    let nproto = r.count()?;
    let mut raw: Vec<RawProto> = Vec::with_capacity(nproto);
    for _ in 0..nproto {
        let size = if version >= 12 { Some(r.varint()? as usize) } else { None };
        let start = r.pos;

        let max_stack = r.u8()?;
        let num_params = r.u8()?;
        let num_upvalues = r.u8()?;
        let is_vararg = r.u8()?;
        let mut flags = 0;
        if version >= 4 {
            flags = r.u8()?;
            if types_version >= 1 {
                let n = r.count()?;
                r.take(n)?;
            }
        }

        let ncode = r.count()?;
        let mut instructions = Vec::with_capacity(ncode);
        for _ in 0..ncode {
            instructions.push(r.u32()?);
        }

        let nk = r.count()?;
        let mut constants: Vec<LuaConstant> = Vec::with_capacity(nk);
        let mut closure_consts = Vec::new();
        for j in 0..nk {
            let k = match r.u8()? {
                LBC_CONSTANT_NIL => LuaConstant::Null,
                LBC_CONSTANT_BOOLEAN => LuaConstant::Bool(r.u8()? != 0),
                LBC_CONSTANT_NUMBER => LuaConstant::Number(LuaNumber::Float(r.f64()?)),
                LBC_CONSTANT_VECTOR => {
                    let v = [r.f32()? as f64, r.f32()? as f64, r.f32()? as f64, r.f32()? as f64];
                    LuaConstant::Vector(v)
                }
                LBC_CONSTANT_VECTORD => LuaConstant::Vector([r.f64()?, r.f64()?, r.f64()?, r.f64()?]),
                LBC_CONSTANT_STRING => match r.string(&strings)? {
                    Some(s) => LuaConstant::String(s),
                    None => return r.err("NULL string constant"),
                },
                // The import path is repeated in GETIMPORT's AUX word.
                LBC_CONSTANT_IMPORT => {
                    r.u32()?;
                    LuaConstant::Null
                }
                LBC_CONSTANT_TABLE => {
                    let n = r.count()?;
                    let mut t = ConstTable::default();
                    for _ in 0..n {
                        let key = r.varint()? as usize;
                        t.hash.push((constants.get(key).cloned().unwrap_or_default(), LuaConstant::Null));
                    }
                    LuaConstant::Table(Box::new(t))
                }
                LBC_CONSTANT_TABLE_WITH_CONSTANTS => {
                    let n = r.count()?;
                    let mut t = ConstTable::default();
                    for _ in 0..n {
                        let key = r.varint()? as usize;
                        let ci = r.u32()? as i32;
                        let value = if ci >= 0 { constants.get(ci as usize).cloned().unwrap_or_default() } else { LuaConstant::Null };
                        t.hash.push((constants.get(key).cloned().unwrap_or_default(), value));
                    }
                    LuaConstant::Table(Box::new(t))
                }
                LBC_CONSTANT_CLOSURE => {
                    closure_consts.push((j, r.varint()? as usize));
                    LuaConstant::Null
                }
                LBC_CONSTANT_INTEGER => {
                    let neg = r.u8()? != 0;
                    let m = r.varint()?;
                    LuaConstant::Number(LuaNumber::Integer(if neg { (!m).wrapping_add(1) as i64 } else { m as i64 }))
                }
                LBC_CONSTANT_CLASS_SHAPE => {
                    r.varint()?;
                    let private_fields = r.u8()? as u64;
                    let private_methods = r.u8()? as u64;
                    let props = r.varint()?;
                    let methods = r.varint()?;
                    // Public field names, then public method names.
                    let named = props.saturating_sub(private_fields) + methods.saturating_sub(private_methods);
                    for _ in 0..named {
                        r.varint()?;
                    }
                    LuaConstant::Null
                }
                _ => return r.err("unknown constant type"),
            };
            constants.push(k);
        }

        let nchild = r.count()?;
        let children = (0..nchild).map(|_| r.varint().map(|v| v as usize)).collect::<R<Vec<_>>>()?;

        let line_defined = r.varint()?;
        let name = r.string(&strings)?.map(|s| s.to_vec()).unwrap_or_default();

        if r.u8()? != 0 {
            let gap = r.u8()?;
            if gap > 31 || ncode == 0 {
                return r.err("bad line info");
            }
            let intervals = ((ncode - 1) >> gap) + 1;
            r.take(ncode)?;
            r.take(intervals * 4)?;
        }

        let mut locals = Vec::new();
        let mut upvalue_names = Vec::new();
        if r.u8()? != 0 {
            let nloc = r.count()?;
            for _ in 0..nloc {
                let name = r.string(&strings)?.map(|s| String::from_utf8_lossy(&s).into_owned()).unwrap_or_default();
                let start_pc = r.varint()?;
                let end_pc = r.varint()?;
                let reg = r.u8()?;
                locals.push(LuaLocal { name, start_pc, end_pc, reg });
            }
            let nup = r.count()?;
            for _ in 0..nup {
                upvalue_names.push(r.string(&strings)?.map(|s| s.to_vec()).unwrap_or_default());
            }
        }

        if version >= 11 {
            let n = r.count()?;
            for _ in 0..n {
                r.u8()?;
                r.varint()?;
            }
        }
        if version >= 12 && flags & LPF_INLINABLE != 0 {
            r.varint()?;
        }
        // Skip unknown trailing data (version 12+ records the proto size).
        if let Some(size) = size {
            match start.checked_add(size) {
                Some(end) if end >= r.pos && end <= input.len() => r.pos = end,
                _ => return r.err("bad proto size"),
            }
        }

        raw.push(RawProto {
            chunk: LuaChunk {
                name,
                line_defined,
                num_upvalues,
                num_params,
                max_stack,
                flags,
                is_vararg: if is_vararg != 0 { Some(LuaVarArgInfo::new()) } else { None },
                instructions,
                constants,
                locals,
                upvalue_names,
                ..Default::default()
            },
            children,
            closure_consts,
        });
    }
    let main = r.varint()? as usize;
    if main >= raw.len() {
        return r.err("main proto index out of range");
    }

    // Build the tree from the main proto.  With inlining (-O2) a proto can be
    // the child of several functions: it is built once and copied.
    #[derive(Clone)]
    enum State { New, Building, Done(Box<LuaChunk>) }
    let mut state = vec![State::New; raw.len()];
    fn build(id: usize, raw: &mut Vec<RawProto>, state: &mut Vec<State>, depth: usize) -> R<LuaChunk> {
        match &state[id] {
            State::Done(c) => return Ok((**c).clone()),
            State::Building => return Err("luau bytecode: proto contains itself".into()),
            State::New if depth > 200 => return Err("luau bytecode: functions nested too deeply".into()),
            State::New => {}
        }
        state[id] = State::Building;
        let children = std::mem::take(&mut raw[id].children);
        let consts = std::mem::take(&mut raw[id].closure_consts);
        let mut chunk = std::mem::take(&mut raw[id].chunk);
        for &(slot, fid) in &consts {
            // Closure constants name a child by global id.
            if let Some(pos) = children.iter().position(|&c| c == fid) {
                chunk.constants[slot] = LuaConstant::Proto(pos);
            }
        }
        for &c in &children {
            if c >= raw.len() {
                return Err("luau bytecode: child proto index out of range".into());
            }
            let child = build(c, raw, state, depth + 1)?;
            chunk.prototypes.push(child);
        }
        state[id] = State::Done(Box::new(chunk.clone()));
        Ok(chunk)
    }
    let main_chunk = build(main, &mut raw, &mut state, 0)?;

    let header = LuaHeader {
        lua_version: LUAU.0,
        format_version: version,
        number_size: 8,
        instruction_size: 4,
        ..Default::default()
    };
    Ok(LuaBytecode { header, main_chunk })
}

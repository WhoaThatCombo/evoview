//! A minimal protobuf reader.
//!
//! Every AC EVO asset - mesh, material, design, animation - is protobuf with no
//! schema shipped, so this walks fields by number and wire type and lets the
//! caller decide what they mean. Field meanings live in load.rs next to the
//! code that uses them.
//!
//! ⚠ Every read is bounds-checked and a malformed field ends the walk rather
//! than panicking. These walkers get pointed at arbitrary bytes - a material's
//! nested blobs are not always protobuf - and one bad length must not take down
//! the load.

#[derive(Debug, Clone, Copy)]
pub enum Val {
    Varint(u64),
    F32(f32),
    F64(f64),
    /// byte range into the buffer the walk started from
    Bytes(usize, usize),
}

impl Val {
    pub fn f32(self) -> f32 {
        match self {
            Val::F32(v) => v,
            Val::Varint(v) => v as f32,
            Val::F64(v) => v as f32,
            _ => 0.0,
        }
    }
    pub fn range(self) -> Option<(usize, usize)> {
        match self {
            Val::Bytes(a, b) => Some((a, b)),
            _ => None,
        }
    }
}

fn varint(d: &[u8], i: &mut usize) -> Option<u64> {
    let mut r = 0u64;
    let mut s = 0u32;
    loop {
        let b = *d.get(*i)?;
        *i += 1;
        r |= ((b & 0x7F) as u64) << s;
        if b & 0x80 == 0 {
            return Some(r);
        }
        s += 7;
        if s > 63 {
            return None;
        }
    }
}

/// Walk the fields in `d[start..end]`, calling `f(field_number, value)`.
pub fn fields<F: FnMut(u32, Val)>(d: &[u8], start: usize, end: usize, mut f: F) {
    let end = end.min(d.len());
    let mut i = start;
    while i < end {
        let key = match varint(d, &mut i) {
            Some(k) => k,
            None => return,
        };
        let (fnum, wt) = ((key >> 3) as u32, (key & 7) as u8);
        match wt {
            0 => match varint(d, &mut i) {
                Some(v) => f(fnum, Val::Varint(v)),
                None => return,
            },
            1 => {
                if i + 8 > end {
                    return;
                }
                let v = f64::from_le_bytes(d[i..i + 8].try_into().unwrap());
                i += 8;
                f(fnum, Val::F64(v));
            }
            2 => {
                let n = match varint(d, &mut i) {
                    Some(n) => n as usize,
                    None => return,
                };
                if i + n > end {
                    return;
                }
                f(fnum, Val::Bytes(i, i + n));
                i += n;
            }
            5 => {
                if i + 4 > end {
                    return;
                }
                let v = f32::from_le_bytes(d[i..i + 4].try_into().unwrap());
                i += 4;
                f(fnum, Val::F32(v));
            }
            _ => return,
        }
    }
}

pub fn text(d: &[u8], r: (usize, usize)) -> String {
    String::from_utf8_lossy(&d[r.0..r.1]).into_owned()
}

/// Packed varints, as used for mesh index buffers.
pub fn varints(d: &[u8], start: usize, end: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = start;
    while i < end.min(d.len()) {
        match varint(d, &mut i) {
            Some(v) => out.push(v as u32),
            None => break,
        }
    }
    out
}

/// A 4x4 row-major matrix written as floats 1..16 with proto3 defaults
/// omitted, which is how both bone rest poses and animation keyframes are
/// stored. Missing fields are zero, so the identity has to be supplied by the
/// caller if that is what an empty message should mean.
pub fn matrix(d: &[u8], start: usize, end: usize) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    fields(d, start, end, |fnum, v| {
        if (1..=16).contains(&fnum) {
            m[(fnum - 1) as usize] = v.f32();
        }
    });
    m
}
